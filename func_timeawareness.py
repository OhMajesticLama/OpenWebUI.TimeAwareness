"""
title: Time Awareness
author: ohmajesticlama
author_url: https://github.com/OhMajesticLama/OpenWebUI.TimeAwareness
funding_url: https://github.com/open-webui
version: 0.1
license: MIT

A function to give time awareness to your conversations.

This inject current time as a system message before each user message.
"""

import time
import sys
import datetime
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, List, Dict
import logging
import functools
import inspect
import json
import asyncio
import zoneinfo

import aiohttp
from fastapi.requests import Request

import open_webui
import open_webui.main
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    query_memory,
    QueryMemoryForm,
    delete_memory_by_id,
)
from open_webui.models.users import Users, User
from open_webui.env import GLOBAL_LOG_LEVEL


# from open_webui.main import webui_app
LOGGER: logging.Logger = logging.getLogger("FUNC:TIME_AWARENESS")


def set_logs(logger: logging.Logger, level: int, force: bool = False):
    """
    logger:
        Logger that will be configured and connected to handlers.

    level:
        Log level per logging module.

    force:
        If set to True, will create and attach a StreamHandler to logger, even if there is already one attached.
    """
    logger.setLevel(level)

    logger.debug("%s has %s handlers", logger, len(logger.handlers))
    for handler in logger.handlers:
        if not force and isinstance(handler, logging.StreamHandler):
            # There is already a stream handler attached to this logger, chances are we don"t want to add another one.
            # This might be a reimport.
            # However we still enforce log level as that's likely what the user expects.
            handler.setLevel(level)
            logger.info("logger already has a StreamHandler. Not creating a new one.")
            return
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(levelname)s[%(name)s]%(lineno)s:%(asctime)s: %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


set_logs(LOGGER, GLOBAL_LOG_LEVEL)
# set_logs(LOGGER, logging.DEBUG)


def log_exceptions(func: Callable[Any, Any]):
    """
    Log exception in decorated function. Use LOGGER of this module.

    Usage:
        @log_exceptions
        def foo():
            ...
            raise Exception()

    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                LOGGER.error("Error in %s: %s", func, exc, exc_info=True)
                raise exc

    else:

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                LOGGER.error("Error in %s: %s", func, exc, exc_info=True)
                raise exc

    return _wrapper


class ROLE:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Filter:
    class Valves(BaseModel):
        timezone: str = Field(
            default_factory=lambda: time.tzname[0],
            description="Default timezone, unless it's overwritten by User Valve. Defaults to system timezone."
            "Timezone must be in: {}".format(
                ", ".join(sorted(zoneinfo.available_timezones()))
            ),
        )

    class UserValves(BaseModel):
        timezone: str = Field(
            default_factory=lambda: time.tzname[0],
            description="User timezone. Overrides Function global timezone if provided. "
            "Timezone must be in: {}".format(
                ", ".join(sorted(zoneinfo.available_timezones()))
            ),
        )
        enabled: bool = Field(
            default=True, description="Enable or disable the time awareness function."
        )

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        self.uservalves = self.UserValves()

    @log_exceptions
    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        if not __user__["valves"].enabled:
            # user doesn't want this, do nothing.
            LOGGER.debug("UserValve.enabled = False. Do nothing.")
            return body
        # Modify the request body or validate it before processing by the chat completion API.
        # This function is the pre-processor for the API where various checks on the input can be performed.
        # It can also modify the request before sending it to the API.
        LOGGER.debug(f"inlet:{__name__}")
        LOGGER.debug(f"inlet:body:{body}")
        LOGGER.debug(f"inlet:user:{__user__}")

        if not "id" in __user__:
            LOGGER.warn("No 'id' key in __user__. Do nothing.")
            return body
        user = Users.get_user_by_id(__user__["id"])

        messages: Optional[Dict[str, str]] = body.get("messages")
        if not messages:
            # nothing to do here.
            return body

        tz = self.valves.timezone
        comment = "This is system timezone. User may not be in this timezone."
        if __user__ is not None:
            try:
                usertz = __user__["valves"].timezone.strip()
                if len(usertz) > 0:
                    tz = usertz
                    comment = "This timezone was provided by the user."
            except KeyError as exc:
                LOGGER.debug("No valves in __user__: %s", __user__)

        if tz not in zoneinfo.available_timezones():
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Unknown timezone: {tz}.",
                        "done": False,
                    },
                }
            )
        now = datetime.datetime.now(tz=zoneinfo.ZoneInfo(tz)).isoformat()
        context = f'<context source="function_time_awareness">\n  <time timezone="{tz}"><!-- {comment} -->{now}</time>\n</context>'

        # Inject context prior to last user message.
        if messages[-1].get("role") == ROLE.USER:
            # Let's insert a system message with context before the user.
            LOGGER.debug("Added context message: %s", context)
            context_message = {"role": ROLE.SYSTEM, "content": context}
            messages.insert(-1, context_message)
        else:
            LOGGER.debug(
                "Last message not from user, do nothing: %s", messages[-1].get("role")
            )

        return body


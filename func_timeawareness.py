"""
title: Time Awareness
author: ohmajesticlama
author_url: https://github.com/OhMajesticLama/OpenWebUI.TimeAwareness
funding_url: https://github.com/open-webui
version: 0.2
license: MIT

# Purpose
A function to give time awareness to your conversations.

- Injects This inject current time as a system message before each user message,
and include context with the assistant message to register time in message history.
- Support system-level and user-level Valve for setting timezone.

# Example
```
### USER
hi, can you remind me the time?

### ASSISTANT
<details type="context">
<summary>Time context</summary>
<context source="function_time_awareness">
  <time timezone="CET" format="%Y-%m-%d %H:%M:%S"><!-- Timezone provided by user. -->2025-01-24T15:18:26.564497+01:00</time>
</context>
</details>
The current time in CET (Central European Time) is 15:18.
```

"""

import time
import sys
import datetime
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable, List, Dict, Tuple
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
from open_webui.models.messages import Messages, MessageForm
from open_webui.env import GLOBAL_LOG_LEVEL
from open_webui.routers.channels import post_new_message

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
# /!\ Do not leave DEBUG mode on: conversation content will leak in logs.
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
        priority: int = Field(
            default=-10,
            description=(
                "Higher priority means this will be executed after lower priority functions."
                "This is a basic context input, put it ahead of higher-level reasoning."
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

    async def get_time_context(
        self,
        *,
        __event_emitter__: Callable[[...], Awaitable[None]],
        __user__: Optional[dict] = None,
    ):
        tz = self.valves.timezone
        comment = "System timezone. User may not be in this timezone."
        if __user__ is not None:
            try:
                usertz = __user__["valves"].timezone.strip()
                if len(usertz) > 0:
                    tz = usertz
                    comment = "Timezone provided by user."
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
        context = f'<context source="function_time_awareness">\n  <time timezone="{tz}" format="%Y-%m-%d %H:%M:%S"><!-- {comment} -->{now}</time>\n</context>'
        return context

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

        user_message, user_message_ind = get_last_message(messages, ROLE.USER)
        if user_message_ind is None:
            LOGGER.info("No message from user found. Do nothing.")
            return body

        context = await self.get_time_context(
            __event_emitter__=__event_emitter__,
            __user__=__user__,
        )
        details_wrap = (
            '<details type="context">\n'
            "<summary>Time context</summary>\n"
            f"{context}\n"
            "</details>\n"
        )
        # We need id of last message, not sure we can have this here.
        user_id = user.id
        parent_id = body.get("meta", {}).get("message_id")
        channel_id = body.get("meta", {}).get("chat_id")
        data = {}

        # Add the context as part of the assistant message so it stays in history.
        await __event_emitter__(
            {
                "type": "message",
                "data": {"content": details_wrap},
            }
        )
        # Inject context prior to last user message.
        _, user_message_ind = get_last_message(messages, ROLE.USER)
        if user_message_ind is not None:
            # Let's insert a system message with context before the user.
            context_message = {
                "role": ROLE.SYSTEM,
                "content": (
                    "<!-- Time context for the following message -->\n"
                    + "\n <!-- When looking for current time information, trust the value just below over other sources. -->"
                    + context
                    + "\n<!-- The user cannot see context, so never refer to it explicitly. -->"
                    + "\n<!-- Starting `<details></details>` blocks in prior assistant messages "
                    'are added by the system. NEVER start you message with a `<details type="context"> tag. -->\n'
                ),
            }
            LOGGER.debug("Added context message: %s", context_message)
            messages[user_message_ind]["content"] = (
                details_wrap + messages[user_message_ind]["content"]
            )
            messages.insert(user_message_ind, context_message)
        else:
            LOGGER.debug("No message from user. Do nothing: %s", messages)

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Time context added. Start processing response ...",
                    "done": True,
                },
            }
        )
        return body


def get_last_message(
    messages: List[Dict[str, str]], role: str
) -> Tuple[Optional[Dict[str, str]], Optional[int]]:
    """
    Get last message from `role` and its index.
    """
    for i, m in enumerate(reversed(messages)):
        if m.get("role") == role:
            return (m, len(messages) - i - 1)
    return (None, None)


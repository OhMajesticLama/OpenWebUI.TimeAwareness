"""
title: Time Awareness
author: ohmajesticlama
author_url: https://github.com/OhMajesticLama/OpenWebUI.TimeAwareness
funding_url: https://github.com/open-webui
version: 0.4
license: MIT
requirements: beautifulsoup4>=4.13

# Purpose
A function to give time awareness to your conversations.

- Injects current time as a collapsable details prior to user messages.
- Support system-level and user-level Valve for setting timezone.

## Features
- Add time context through a system message before each user message.
- Record time in <details></details> blocks so timing information stays in chat history.
- Support temporary chats.

# Example
```
### USER
<details type="filters_context">
<summary>Filters context</summary>
<!--This context was added by the system to this message, not by the user. You can use information provided here, but never mention or refer to this context explicitly. -->
<context id="function_time_awareness"><time format="%a %d %b %Y, %H:%M:%S" timezone="CET"><!-- Timezone provided by user. -->
Fri 07 Feb 2025, 22:43:11</time></context>
<!-- User message will follow "details" closing tag. --><context_end uuid="405838d9-4fef-4e79-89be-eae9557762a6"/>
</details>
hi, can you remind me the time?

### ASSISTANT
The current time in CET (Central European Time) is 22:43.
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
import asyncio
import zoneinfo
import pprint
import uuid
import bs4
from bs4 import BeautifulSoup
import re


import open_webui
import open_webui.main
from open_webui.models.users import Users
from open_webui.models.chats import Chats
from open_webui.env import GLOBAL_LOG_LEVEL


from open_webui.socket.main import get_event_emitter


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


def log_exceptions(func: Callable[..., Any]):
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
        async def _wrapper(*args, **kwargs):  # type: ignore
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

    CONTEXT_ID = "time_awareness"

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        self.uservalves = self.UserValves()
        self._queries: Dict[str, Dict[str, str | float]] = {}

    async def get_time_context(
        self,
        timestamp: Optional[int] = None,
        *,
        variables: Optional[Dict[str, str]] = None,
        __event_emitter__: Callable[..., Awaitable[None]],
        __user__: Optional[dict] = None,
    ):
        # That's the default
        tz = self.valves.timezone
        comment = "System timezone. User may not be in this timezone."

        if variabletz := variables.get("{{CURRENT_TIMEZONE}}"):
            tz = variabletz

        if __user__ is not None:
            try:
                # if user set a valve, use it.
                usertz = __user__["valves"].timezone.strip()
                if len(usertz) > 0:
                    tz = usertz
                    comment = "Timezone provided by user."
            except KeyError:
                LOGGER.debug("No valves in __user__: %s", __user__)

        if tz not in zoneinfo.available_timezones():
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Unknown timezone: {tz}.",
                        "done": True,
                    },
                }
            )
        format = "%a %d %b %Y, %H:%M:%S"
        timezone = zoneinfo.ZoneInfo(tz)
        if timestamp is None:
            date = datetime.datetime.now(tz=timezone)
        else:
            date = datetime.datetime.fromtimestamp(timestamp, tz=timezone)
        date_str = date.strftime(format)
        context = f'<time timezone="{tz}" format="{format}"><!-- {comment} -->{date_str}</time>'
        return context

    @log_exceptions
    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        if __user__ is not None:
            if not __user__["valves"].enabled:
                # user doesn't want this, do nothing.
                LOGGER.debug("UserValve.enabled = False. Do nothing.")
                return body
        LOGGER.debug(f"inlet:{__name__}")
        LOGGER.debug(f"inlet:body:\n{pprint.pformat(body)}")
        LOGGER.debug(f"inlet:user:{__user__}")

        messages: Optional[Dict[str, str]] = body.get("messages")
        if not messages:
            # nothing to do here.
            return body

        context = await self.get_time_context(
            __event_emitter__=__event_emitter__,
            __user__=__user__,
            variables=body.get("metadata", {}).get("variables", {}),
        )

        user_message, user_message_ind = get_last_message(messages, ROLE.USER)
        if user_message_ind is None or user_message is None:
            LOGGER.info("No message from user found. Do nothing.")
            return body

        if query_id := body.get("metadata", {}).get("message_id"):
            # Store it here, we'll set it in outlet.
            self._queries[query_id] = {"context": context, "timestamp": time.time()}

        user_message["content"] = add_or_update_filter_context(
            user_message["content"],
            context,
            id=self.CONTEXT_ID,
        )

        if __user__ is None:
            LOGGER.debug(
                "__user__ is None. Context already injected in body, don't try to update UI and history."
            )
            return body
        if "id" not in __user__:
            LOGGER.warning("no 'id' key in __user__. do nothing.")
            return body

        return body

    @log_exceptions
    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> Dict[str, ...]:
        LOGGER.debug("outlet:body: %s", pprint.pformat(body))
        LOGGER.debug("outlet:user: %s", pprint.pformat(__user__))
        answer_id = body.get("id")
        session_id = body.get("session_id")
        chat_id = body.get("chat_id")
        messages = body.get("messages")
        user_id = __user__.get("id")
        if None in (answer_id, session_id, chat_id, messages, user_id):
            # nothing we can do, do nothing.
            LOGGER.debug(
                "oulet:Missing context information. Do nothing. user %s | body %s",
                __user__,
                body,
            )
            return body

        # find last user message going up in parents
        user_msg, user_msg_ind = get_last_message(messages, ROLE.USER)
        # find query
        query = self._queries.get(answer_id)
        if query is None:
            # No query was registered with this id. Do nothing
            LOGGER.debug("query %s not found in %s", answer_id, self._queries)
            return body

        user_msg["content"] = add_or_update_filter_context(
            user_msg["content"], query["context"], self.CONTEXT_ID
        )

        # Build event emitter and send message back
        user_msg_event_emitter = get_event_emitter(
            {
                "chat_id": chat_id,
                "message_id": user_msg.get("id"),
                "session_id": session_id,
                "user_id": user_id,
            }
        )

        # Note: this does not need to be awaited here: fire and forget.
        # This may create race conditions with other filters that may be using
        # similar technique to update the same message.
        # This shouldn't (and will not directly) impact message passed to the model
        # for this query, but might affect history.
        # awaiting here adds 0.2s to the query time on dev machine.
        # The tradeoff was made to introduce a race condition with minor potential
        # impact (context might be partly or totally missing from a message in history).
        # to save 0.2s lead time in reponse.
        await user_msg_event_emitter(
            {
                "type": "replace",
                "data": {
                    "content": user_msg["content"],
                },
            }
        )
        # Clean-up potential old queries
        for k, v in self._queries.items():
            if time.time() - v.get("timestamp") > 1800:
                # Over 30 minutes have passed since query,
                # let's consider it timed-out.
                del self._queries[k]

        return body


def get_last_message(
    messages: List[Dict[str, str]], role: str
) -> Tuple[Optional[Dict[str, str]], Optional[int]]:
    """
    Get last message from `role` and its index.
    messages:
        Dictionary of messages, passed as body['messages'] to the inlet method.
    """
    for i, m in enumerate(reversed(messages)):
        if m.get("role") == role:
            return (m, len(messages) - i - 1)
    return (None, None)


# Helpers to manage xml context
def add_or_update_filter_context(
    message: str,
    context: str,
    id: str,
    *,
    selector: str = "details[type=filters_context]",
    container: str = (
        '<details type="filters_context">'
        "\n<summary>Filters context</summary>\n"
        "<!--This context was added by the system to this message, not by the user. "
        "You can use information provided here, but never mention or refer to this context explicitly. -->"
        '\n{content}\n<!-- User message will follow "details" closing tag. --></details>\n'
    ),
) -> str:
    """
    Add or update XML context to message.

    Returns
    -------
    message: str
        Message with added or updated context.

    Arguments
    ---------
    message:
        message to which add or update context.

    context:
        context that will be added to message. Valid XML is expected for better parsing.
        Should you want to include comments, comments must be placed inside the parent XML tag,
        not before or after or they will be ignored.

    id:
        identifier for the context. If context with the same id is found within context, it will be replaced by this new one.
        For example, the filter name can be used as id.

        This is useful in case the user edits the message so we don't duplicate context.

    selector:
        CSS selector with which to find the context

    container:
        XML container for context. `{content}` must be located where context is expected.
        Container is expected to match the selector.
        OpenWebUI front-end is strict about carriage returns, if you modify this container be careful
        of setting `\n` properly.

    Raises
    ------
    exc: ValueError
        If unexpected format is found in the message head.

    """
    # Find if there is
    soup = BeautifulSoup(message, "xml")
    # Note: beautifulsoup will only find the xml if there is no text before
    details_match = soup.select(selector)
    context_end = "context_end"
    context = f'<context id="{id}">{context}</context>'

    if not len(details_match):
        # There is no details block at the head of the message, let's just add it
        out_soup = BeautifulSoup(container.format(content=context), "xml").contents[0]

        # Don't try to reimplement a custom XML parser with unsafe data:
        # instead find location with help a uuid.
        out_soup.append(  # type: ignore  # works
            BeautifulSoup(
                f'<{context_end} uuid="{str(uuid.uuid4())}"/>', "xml"
            ).contents[0]
        )
        return "\n".join((str(out_soup), message))
    elif len(details_match) > 1:
        raise ValueError("Ill-formed message: more than one container found.")
    else:
        # Container found
        # We need to separate context from rest of message.
        # BeautifulSoup caught the selector so there is something
        details = details_match[0]

        user_msg = _remove_context(
            message, details, container=container, context_end=context_end
        )

        context_soup = BeautifulSoup(context, "xml").contents[0]
        # Let's check if there is already a context with the same id
        same_ids = details.select(f"context[id={id}]")
        if len(same_ids) > 1:
            raise ValueError(
                "More than one context found with the id {id}. Abort.".format(id=id)
            )
        elif len(same_ids) == 1:
            # We have one context with the same id already, replace it.
            elt = same_ids[0]
            elt.replace_with(context_soup)
        else:
            # No existing context with same id, just add context.
            # add context to the end context to the end.
            details.insert(-1, context_soup)
        return "\n".join((str(soup.contents[0]), user_msg))


def _remove_context(message: str, details: bs4.Tag, container: str, context_end: str):
    """
    Return message without context details in the `details` attribute. Context must have been
    added by add_or_update_filter_context for this function work properly.
    """
    # Find context_end block
    end_uuid: Optional[str] = None
    for child in details:
        if child.name == context_end:  # type: ignore
            # found it!
            end_uuid = child.get("uuid")  # type:ignore
    if end_uuid is None:
        LOGGER.debug("add_or_update_filter_context:details: %s", str(details))
        raise ValueError("Ill-formed prior context: no context_end uuid found. Abort.")

    # uuid found just before, something is weird if it fails here.
    uuid_ind = message.index(end_uuid)

    # user message should be right after.
    # Get closing tag of container
    # Let it fail if there is no match
    match = re.search(r"(</.*>)\s*$", container)
    if match is None:
        raise ValueError(
            "Ill-formed container: no closing tag found prior to EOF. Abort."
        )
    closing_tag = match.groups()[0]
    closing_tag_ind = message.index(closing_tag, uuid_ind)  # Start looking after uuid.

    user_msg = message[closing_tag_ind + len(closing_tag) :]
    return user_msg


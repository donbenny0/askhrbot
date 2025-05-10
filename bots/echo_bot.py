# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount
from rag.chat import timed_query


class EchoBot(ActivityHandler):
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello how can i help you today?😊")

    async def on_message_activity(self, turn_context: TurnContext):
        query_response=timed_query(turn_context.activity.text)
        return await turn_context.send_activity(
            MessageFactory.text(query_response)
        )

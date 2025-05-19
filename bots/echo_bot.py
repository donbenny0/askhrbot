# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount
from rag.chat import chat


class EchoBot(ActivityHandler):
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello how can i help you today?😊")
                
    def _extract_user_prefix(self, user_id: str) -> str:
        """Extract just the numeric prefix from Teams user ID"""
        if ':' in user_id:
            return user_id.split(':')[1]
        return user_id
    
    async def on_message_activity(self, turn_context: TurnContext):
        # You can log these IDs if needed
        user_id = self._extract_user_prefix(turn_context.activity.recipient.id)
        print(f"User ID: {user_id}")
        query_response=chat(turn_context.activity.text,user_id)
        return await turn_context.send_activity(
            MessageFactory.text(query_response)
        )

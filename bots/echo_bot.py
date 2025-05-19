# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount, ActivityTypes,Activity
from rag.chat import chat


class EchoBot(ActivityHandler):
    async def on_members_added_activity(
        self, members_added: [ChannelAccount], turn_context: TurnContext
    ):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello how can i help you today?😊")
                
    def _extract_user_id(self, user_id: str) -> str:
        """Extract the user ID from Teams user ID format"""
        # Teams user IDs often come in format like "29:1abc..."
        if ':' in user_id:
            return user_id.split(':')[-1]  # Get the last part after colon
        return user_id

    def _extract_conversation_id(self, conv_id: str) -> str:
        """Extract the conversation/session ID"""
        # Teams conversation IDs can be long, you might want to hash them
        return conv_id
    
    async def on_message_activity(self, turn_context: TurnContext):
        # Send typing indicator
        typing_activity = Activity(type=ActivityTypes.typing)
        await turn_context.send_activity(typing_activity)
        
        # Process the message
        user_id = self._extract_user_id(turn_context.activity.from_property.id)
        conversation_id = self._extract_conversation_id(turn_context.activity.conversation.id)
        query_response = chat(turn_context.activity.text, user_id, conversation_id)
        
        return await turn_context.send_activity(
            MessageFactory.text(query_response)
        )
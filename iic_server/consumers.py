import json
from channels.generic.websocket import AsyncWebsocketConsumer

class MyConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("info_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("info_group", self.channel_name)

    def receive(self, message, product, face_name, product_name):
        async_to_sync(self.channel_layer.group_send)(
            self.room_group_name, {"type": "chat_message", "message": message, "product": product, 'face_name': face_name,'product_name': product_name}
        )

    async def send_message(self, event):
        message = event['message']
        product = event['product']
        face_name = event['face_name']
        product_name = event['product_name']
        await self.send(text_data=json.dumps({
            'message': message,
            'product': product,
            'face_name': face_name,
            'product_name': product_name
        }))
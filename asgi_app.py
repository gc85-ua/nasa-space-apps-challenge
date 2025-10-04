from asgiref.wsgi import WsgiToAsgi
from main import app as flask_app

# Convert Flask (WSGI) app to ASGI so uvicorn/uv puede ejecutarla directamente
asgi_app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    # Quick internal test when ejecutado con python
    import asyncio
    async def _test():
        print('ASGI wrapper listo: asgi_app')
    asyncio.run(_test())

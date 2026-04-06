# requirements: numpy, skyfield, sgp4

# main.py
import sys
import asyncio
import traceback
import pygame

pygame.init()

if sys.platform in ["emscripten", "wasm32"]:
    SCREEN = pygame.display.set_mode((1280, 720))
else:
    SCREEN = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)

pygame.display.set_caption("Space Debris Cleaner")

async def main():
    import js
    try:
        js.console.log("0. Display Ready! Yielding to browser...")
        await asyncio.sleep(0.1) 

        js.console.log("1. Loading application modules...")

        from app import SpaceDebrisApp
        
        js.console.log("2. Modules loaded! Initializing Game App...")
        
        app = SpaceDebrisApp(SCREEN)
        
        js.console.log("3. Setup Starry Sky...")
        await app.renderer.async_setup_starry_sky()
        
        js.console.log("4. Starting Main Loop!")
        await app.run()
        
    except Exception as e:
        js.console.error(f"FATAL ERROR IN ASYNC MAIN: {str(e)}")
        err_msg = traceback.format_exc()
        js.console.error(err_msg)
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

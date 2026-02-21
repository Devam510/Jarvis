"""
Jarvis — Autonomous Desktop Agent
Entry point: python -m jarvis
"""

import asyncio
import logging
import signal
import sys

from jarvis.utils.config import JarvisConfig
from jarvis.core.orchestrator import AgentOrchestrator

logger = logging.getLogger("jarvis")


def print_banner():
    print(
        r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
    Autonomous Desktop Agent v0.1.0
    """
    )


async def main():
    print_banner()

    # Load configuration
    config = JarvisConfig.load("config.yaml")

    # Create and start orchestrator
    orchestrator = AgentOrchestrator(config)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()

    def _signal_handler():
        logger.info("Shutdown signal received")
        asyncio.ensure_future(orchestrator.stop())

    try:
        loop.add_signal_handler(signal.SIGINT, _signal_handler)
        loop.add_signal_handler(signal.SIGTERM, _signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
        await orchestrator.stop()
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

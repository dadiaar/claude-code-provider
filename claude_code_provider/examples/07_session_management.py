#!/usr/bin/env python3
# Copyright (c) 2025. Claude Code Provider for Microsoft Agent Framework.

"""Example 07: Session Management

Track, list, and manage conversation sessions.
Sessions persist conversation state and can be exported/restored.

Concepts demonstrated:
    - Creating a SessionManager
    - Tracking sessions with metadata
    - Listing and querying sessions
    - Getting session statistics
    - Cleaning up sessions

Run:
    python -m claude_code_provider.examples.07_session_management
"""

import asyncio
import os
import tempfile

from claude_code_provider import ClaudeCodeClient, SessionManager


async def main() -> None:
    """Run the session management example."""
    print("Example 07: Session Management")
    print("=" * 40)

    # Create session manager with temporary storage
    temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        manager = SessionManager(storage_path=temp_path)
        client = ClaudeCodeClient(model="haiku")

        # Create multiple sessions
        print("\nCreating sessions...")
        sessions = []
        for i in range(3):
            response = await client.get_response(f"Session {i+1}: Say hello")
            session_id = client.current_session_id

            if session_id:
                # Track the session with metadata
                info = manager.track_session(
                    session_id,
                    model="haiku",
                    topic=f"Session {i+1}",
                )
                sessions.append(info)
                print(f"  Created session: {session_id[:20]}...")

            # Reset to start a new session
            client.reset_session()

        # List all sessions
        print("\nAll tracked sessions:")
        for session in manager.list_sessions():
            print(f"  - {session.session_id[:20]}... (model: {session.model})")

        # Get statistics
        stats = manager.get_stats()
        print(f"\nSession Statistics:")
        print(f"  Total sessions: {stats['total_sessions']}")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Models used: {stats['models_used']}")

        # Get recent sessions
        recent = manager.get_recent_sessions(limit=2)
        print(f"\nMost recent {len(recent)} sessions:")
        for s in recent:
            print(f"  - {s.session_id[:20]}...")

        # Cleanup
        print("\nCleaning up sessions...")
        count = manager.clear_all()
        print(f"Cleared {count} sessions")

    finally:
        # Clean up temp file
        os.unlink(temp_path)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

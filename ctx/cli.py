import argparse

from ctx.db import get_connection
from ctx.facts import FactStore, NumpyRandomEmbedder
from ctx.session import SessionManager


def cmd_sessions_list(args: argparse.Namespace) -> None:
    conn = get_connection()
    mgr = SessionManager(conn)
    sessions = mgr.list_sessions()
    if not sessions:
        print("No sessions yet.")
        return
    for s in sessions:
        print(f"{s.id}  {s.created_at:%Y-%m-%d %H:%M}  [{s.domain}]  {s.title}")


def cmd_facts_list(args: argparse.Namespace) -> None:
    conn = get_connection()
    store = FactStore(conn, NumpyRandomEmbedder())
    facts = store.get_all()
    if not facts:
        print("No facts yet.")
        return
    for f in facts:
        conf = f"({f.confidence:.0%})"
        entities = ", ".join(f.entities) if f.entities else ""
        print(f"  {f.id}  [{f.category}] {conf}  {f.claim}  {entities}")


def cmd_sessions_create(args: argparse.Namespace) -> None:
    conn = get_connection()
    mgr = SessionManager(conn)
    session = mgr.create_session(domain=args.domain, title=args.title)
    print(f"Created session {session.id} ({session.title})")


def main():
    parser = argparse.ArgumentParser(
        prog="ctx",
        description="Cascading context for AI agents",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    # sessions list
    sp_sessions = subparsers.add_parser("sessions", help="Manage sessions")
    sessions_sub = sp_sessions.add_subparsers(dest="sessions_command")

    sessions_sub.add_parser("list", help="List all sessions")

    sp_create = sessions_sub.add_parser("create", help="Create a new session")
    sp_create.add_argument("domain", help="Session domain (e.g. 'coding', 'research')")
    sp_create.add_argument("--title", help="Optional session title")

    # facts
    sp_facts = subparsers.add_parser("facts", help="Manage facts")
    facts_sub = sp_facts.add_subparsers(dest="facts_command")
    facts_sub.add_parser("list", help="List all stored facts")

    args = parser.parse_args()

    if args.command == "facts":
        if args.facts_command == "list":
            cmd_facts_list(args)
        else:
            sp_facts.print_help()
    elif args.command == "sessions":
        if args.sessions_command == "list":
            cmd_sessions_list(args)
        elif args.sessions_command == "create":
            cmd_sessions_create(args)
        else:
            sp_sessions.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

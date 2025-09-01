#!/usr/bin/env python3
"""
Utility script to view and analyze debug logs.
Updated version that works with the restructured codebase.
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def list_sessions(log_dir: str = "logs/debug") -> List[str]:
    """List all debug sessions."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return []
    
    sessions = set()
    for file in log_path.glob("*.json"):
        session_id = file.name.split("_msg")[0]
        sessions.add(session_id)
    
    return sorted(list(sessions))

def get_session_files(session_id: str, log_dir: str = "logs/debug") -> List[Path]:
    """Get all files for a session."""
    log_path = Path(log_dir)
    return sorted(log_path.glob(f"{session_id}_*.json"))

def load_log_file(filepath: Path) -> Dict[str, Any]:
    """Load a log file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e), "filepath": str(filepath)}

def print_session_summary(session_id: str, log_dir: str = "logs/debug"):
    """Print summary of a debug session."""
    files = get_session_files(session_id, log_dir)
    if not files:
        print(f"No files found for session: {session_id}")
        return
    
    print(f"\n🔍 Debug Session: {session_id}")
    print("=" * 60)
    
    messages = {}
    
    for file in files:
        data = load_log_file(file)
        if "error" in data and "filepath" in data:
            print(f"❌ Error loading: {file.name}")
            continue
        
        msg_count = data.get("message_count", 0)
        log_type = data.get("type", "unknown")
        
        if msg_count not in messages:
            messages[msg_count] = {}
        messages[msg_count][log_type] = data
    
    for msg_count in sorted(messages.keys()):
        msg_data = messages[msg_count]
        print(f"\n📨 Message {msg_count}:")
        
        # User input
        if "user_input" in msg_data:
            input_data = msg_data["user_input"]
            print(f"   👤 Input: {input_data['message'][:100]}...")
            print(f"       Length: {input_data['message_length']} chars")
        
        # Tool executions
        for log_type, data in msg_data.items():
            if log_type.startswith("tool_"):
                tool_name = data.get("tool_name", log_type.replace("tool_", ""))
                success = "✅" if data.get("success", False) else "❌"
                result_len = data.get("result_length", 0)
                print(f"   🔧 Tool: {tool_name} {success} ({result_len} chars)")
        
        # Agent response
        if "agent_response" in msg_data:
            response_data = msg_data["agent_response"]
            is_empty = response_data.get("is_empty", True)
            response_len = response_data.get("response_length", 0)
            status = "❌ Empty" if is_empty else "✅ Success"
            print(f"   🤖 Response: {status} ({response_len} chars)")
            
            if not is_empty:
                response_preview = response_data.get("response", "")[:100]
                print(f"       Preview: {response_preview}...")
        
        # Runner result details
        if "runner_result" in msg_data:
            result_data = msg_data["runner_result"]
            had_output = result_data.get("has_final_output", False)
            output_len = result_data.get("final_output_length", 0)
            new_items = result_data.get("new_items_info", [])
            
            print(f"   🏃 Runner: {'✅' if had_output else '❌'} final_output ({output_len} chars)")
            print(f"       Items: {len(new_items)} ({', '.join([item.get('type', 'unknown') for item in new_items[:3]])})") 

def print_detailed_log(session_id: str, message_num: int, log_dir: str = "logs/debug"):
    """Print detailed log for a specific message."""
    files = get_session_files(session_id, log_dir)
    
    target_files = [f for f in files if f"_msg{message_num:03d}_" in f.name]
    
    if not target_files:
        print(f"No logs found for session {session_id}, message {message_num}")
        return
    
    print(f"\n📋 Detailed Log - Session: {session_id}, Message: {message_num}")
    print("=" * 80)
    
    for file in sorted(target_files):
        data = load_log_file(file)
        print(f"\n📄 {file.name}")
        print("-" * 40)
        print(json.dumps(data, indent=2, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="View GPT-OSS Agent debug logs")
    parser.add_argument("--list", "-l", action="store_true", help="List all debug sessions")
    parser.add_argument("--session", "-s", type=str, help="Show summary for a specific session")
    parser.add_argument("--detailed", "-d", type=int, help="Show detailed log for a specific message number (requires --session)")
    parser.add_argument("--log-dir", type=str, default="logs/debug", help="Debug logs directory")
    
    args = parser.parse_args()
    
    if args.list:
        sessions = list_sessions(args.log_dir)
        if not sessions:
            print("No debug sessions found.")
        else:
            print("🗂️  Available Debug Sessions:")
            for session in sessions:
                print(f"   📁 {session}")
    
    elif args.session:
        if args.detailed is not None:
            print_detailed_log(args.session, args.detailed, args.log_dir)
        else:
            print_session_summary(args.session, args.log_dir)
    
    else:
        # Show latest session by default
        sessions = list_sessions(args.log_dir)
        if sessions:
            latest = sessions[-1]
            print(f"Showing latest session: {latest}")
            print_session_summary(latest, args.log_dir)
        else:
            print("No debug sessions found. Use --help for options.")

if __name__ == "__main__":
    main()
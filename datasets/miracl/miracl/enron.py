"""Enron email dataset assets for RAG evaluation.

This module provides Dagster assets for:
1. Downloading Enron email dataset from CMU
2. Converting emails to markdown files
3. Organizing emails by user
"""

import email
import json
import re
import tarfile
import urllib.request
from email import policy
from pathlib import Path
from typing import Any

from dagster import asset, AssetExecutionContext, Config


class EnronConfig(Config):
    """Configuration for Enron email dataset assets."""

    output_dir: str = "output/enron"
    # CMU Enron email dataset URL
    dataset_url: str = "http://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    max_emails: int = 0  # 0 means no limit


def _parse_email_file(file_path: Path) -> dict[str, Any] | None:
    """Parse a single email file and extract its contents.
    
    Args:
        file_path: Path to the email file.
        
    Returns:
        Dict containing email metadata and content, or None if parsing fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            msg = email.message_from_file(f, policy=policy.default)
        
        # Extract email headers
        subject = msg.get("Subject", "") or ""
        from_addr = msg.get("From", "") or ""
        to_addr = msg.get("To", "") or ""
        cc_addr = msg.get("Cc", "") or ""
        date = msg.get("Date", "") or ""
        
        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="ignore")
                        break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
            else:
                body = msg.get_payload() or ""
        
        return {
            "subject": subject,
            "from": from_addr,
            "to": to_addr,
            "cc": cc_addr,
            "date": date,
            "body": body.strip(),
            "file_path": str(file_path),
        }
    except Exception:
        return None


@asset(
    description="Download and extract Enron email dataset from CMU",
    group_name="enron",
)
def enron_emails(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Download and extract the Enron email dataset.
    
    The dataset contains approximately 500,000 emails from about 150 Enron employees.
    Source: Carnegie Mellon University CALO Project.
    
    Returns:
        Dict containing dataset metadata and paths.
    """
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_dir = output_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the tar.gz file
    tar_file = cache_dir / "enron_mail.tar.gz"
    
    if not tar_file.exists():
        context.log.info(f"Downloading Enron email dataset from {config.dataset_url}...")
        context.log.info("This may take a while (~400MB)...")
        urllib.request.urlretrieve(config.dataset_url, tar_file)
        context.log.info(f"Downloaded to {tar_file}")
    else:
        context.log.info(f"Using cached file: {tar_file}")
    
    # Extract the tar.gz file
    extract_dir = output_path / "maildir"
    
    if not extract_dir.exists():
        context.log.info("Extracting tar.gz file...")
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=output_path)
        context.log.info(f"Extracted to {output_path}")
    else:
        context.log.info(f"Using extracted data: {extract_dir}")
    
    # Parse all email files
    context.log.info("Parsing email files...")
    
    emails_list: list[dict] = []
    email_count = 0
    
    # Find the maildir (it might be nested)
    if (output_path / "maildir").exists():
        maildir = output_path / "maildir"
    elif (output_path / "enron_mail_20150507" / "maildir").exists():
        maildir = output_path / "enron_mail_20150507" / "maildir"
    else:
        # Search for maildir
        maildir = None
        for item in output_path.rglob("maildir"):
            if item.is_dir():
                maildir = item
                break
        if maildir is None:
            raise FileNotFoundError("Could not find maildir in extracted data")
    
    context.log.info(f"Found maildir at: {maildir}")
    
    # Iterate through all users and their emails
    for user_dir in sorted(maildir.iterdir()):
        if not user_dir.is_dir():
            continue
        
        user_name = user_dir.name
        
        # Iterate through all folders for this user
        for email_file in user_dir.rglob("*"):
            if not email_file.is_file():
                continue
            
            # Skip non-email files
            if email_file.suffix in [".DS_Store", ".gitignore"]:
                continue
            
            email_data = _parse_email_file(email_file)
            if email_data:
                email_data["user"] = user_name
                email_data["folder"] = str(email_file.parent.relative_to(user_dir))
                emails_list.append(email_data)
                email_count += 1
                
                if email_count % 10000 == 0:
                    context.log.info(f"Parsed {email_count} emails...")
                
                if config.max_emails > 0 and email_count >= config.max_emails:
                    context.log.info(f"Reached max_emails limit: {config.max_emails}")
                    break
        
        if config.max_emails > 0 and email_count >= config.max_emails:
            break
    
    context.log.info(f"Total emails parsed: {len(emails_list)}")
    
    # Save emails to JSONL
    emails_file = output_path / "emails.jsonl"
    with open(emails_file, "w", encoding="utf-8") as f:
        for email_data in emails_list:
            f.write(json.dumps(email_data, ensure_ascii=False) + "\n")
    
    context.log.info(f"Saved emails to {emails_file}")
    
    # Get unique users
    unique_users = set(e["user"] for e in emails_list)
    
    return {
        "email_count": len(emails_list),
        "unique_users": len(unique_users),
        "emails_path": str(emails_file),
        "maildir_path": str(maildir),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files",
    group_name="enron",
)
def enron_markdown_docs(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to individual markdown files.
    
    Each email is saved as a markdown file with:
    - Subject as H1 header
    - Metadata (From, To, CC, Date) as a header section
    - Email body as content
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_docs"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown in {md_output_dir}")
    
    doc_count = 0
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "")
            folder = email_data.get("folder", "")
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                f"| User | {user} |",
                f"| Folder | {folder} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use doc_count as filename to ensure uniqueness
            doc_id = f"{doc_count:06d}"
            md_file = md_output_dir / f"{doc_id}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


@asset(
    deps=[enron_emails],
    description="Convert Enron emails to markdown files grouped by user",
    group_name="enron",
)
def enron_markdown_by_user(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Convert Enron emails to markdown files, organized by user.
    
    Creates a directory structure:
    - {user}/
      - {folder}/
        - {email_id}.md
    
    Returns:
        Dict containing markdown output metadata.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    md_output_dir = Path(config.output_dir) / "markdown_by_user"
    md_output_dir.mkdir(parents=True, exist_ok=True)
    
    context.log.info(f"Converting emails to markdown by user in {md_output_dir}")
    
    doc_count = 0
    user_email_counts: dict[str, int] = {}
    doc_index = {}
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            
            subject = email_data.get("subject", "") or "No Subject"
            from_addr = email_data.get("from", "")
            to_addr = email_data.get("to", "")
            cc_addr = email_data.get("cc", "")
            date = email_data.get("date", "")
            body = email_data.get("body", "")
            user = email_data.get("user", "unknown")
            folder = email_data.get("folder", "inbox")
            
            # Create user/folder directory
            user_dir = md_output_dir / user / folder
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # Track email count per user for unique naming
            if user not in user_email_counts:
                user_email_counts[user] = 0
            user_email_counts[user] += 1
            email_num = user_email_counts[user]
            
            # Create markdown content
            md_lines = [
                f"# {subject}",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| From | {from_addr} |",
                f"| To | {to_addr} |",
            ]
            
            if cc_addr:
                md_lines.append(f"| CC | {cc_addr} |")
            
            md_lines.extend([
                f"| Date | {date} |",
                "",
                "---",
                "",
                body,
            ])
            
            md_content = "\n".join(md_lines) + "\n"
            
            # Use email number as filename
            doc_id = f"{user}/{folder}/{email_num:05d}"
            md_file = user_dir / f"{email_num:05d}.md"
            
            with open(md_file, "w", encoding="utf-8") as mf:
                mf.write(md_content)
            
            doc_index[doc_id] = {
                "path": str(md_file),
                "subject": subject,
                "user": user,
                "folder": folder,
            }
            doc_count += 1
            
            if doc_count % 10000 == 0:
                context.log.info(f"Wrote {doc_count} markdown files...")
    
    # Save document index
    index_file = md_output_dir / "doc_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)
    
    context.log.info(f"Converted {doc_count} emails to markdown")
    context.log.info(f"Unique users: {len(user_email_counts)}")
    context.log.info(f"Document index saved to {index_file}")
    
    return {
        "document_count": doc_count,
        "unique_users": len(user_email_counts),
        "markdown_dir": str(md_output_dir),
        "index_path": str(index_file),
    }


def _extract_email_addresses(email_field: str) -> list[str]:
    """Extract all email addresses from an email header field.
    
    Args:
        email_field: Email header field (e.g., "John Doe <john@enron.com>, jane@example.com")
        
    Returns:
        List of extracted email addresses in lowercase.
    """
    # Match email addresses in various formats
    pattern = r'[\w\.-]+@[\w\.-]+'
    addresses = re.findall(pattern, email_field.lower())
    return addresses


def _is_enron_email(address: str) -> bool:
    """Check if an email address is from Enron domain.
    
    Args:
        address: Email address to check.
        
    Returns:
        True if the address is from @enron.com domain.
    """
    return address.lower().endswith('@enron.com')


@asset(
    deps=[enron_emails],
    description="Separate Enron emails into internal (Enron) and external emails",
    group_name="enron",
)
def enron_email_separation(context: AssetExecutionContext, config: EnronConfig) -> dict[str, Any]:
    """Separate emails into Enron internal and external categories.
    
    Classification logic:
    - Enron emails: Emails where the sender (From) has an @enron.com address
    - External emails: Emails where the sender (From) does NOT have an @enron.com address
    
    Outputs:
    - enron_emails.json: All emails from Enron employees
    - external_emails.json: All emails from external senders
    
    Returns:
        Dict containing separation statistics and file paths.
    """
    emails_path = Path(config.output_dir) / "emails.jsonl"
    
    if not emails_path.exists():
        raise FileNotFoundError(f"Emails file not found: {emails_path}. Run enron_emails first.")
    
    context.log.info(f"Reading emails from {emails_path}")
    
    enron_internal: list[dict] = []
    external: list[dict] = []
    
    with open(emails_path, "r", encoding="utf-8") as f:
        for line in f:
            email_data = json.loads(line)
            from_field = email_data.get("from", "")
            
            # Extract sender email addresses
            sender_addresses = _extract_email_addresses(from_field)
            
            # Check if any sender address is from Enron
            is_from_enron = any(_is_enron_email(addr) for addr in sender_addresses)
            
            if is_from_enron:
                enron_internal.append(email_data)
            else:
                external.append(email_data)
    
    total_count = len(enron_internal) + len(external)
    context.log.info(f"Total emails processed: {total_count}")
    context.log.info(f"Enron internal emails: {len(enron_internal)}")
    context.log.info(f"External emails: {len(external)}")
    
    # Save Enron internal emails
    enron_file = Path(config.output_dir) / "enron_emails.json"
    with open(enron_file, "w", encoding="utf-8") as f:
        json.dump(enron_internal, f, ensure_ascii=False, indent=2)
    context.log.info(f"Saved Enron emails to {enron_file}")
    
    # Save external emails
    external_file = Path(config.output_dir) / "external_emails.json"
    with open(external_file, "w", encoding="utf-8") as f:
        json.dump(external, f, ensure_ascii=False, indent=2)
    context.log.info(f"Saved external emails to {external_file}")
    
    # Collect some statistics about senders
    enron_senders: set[str] = set()
    external_senders: set[str] = set()
    
    for email_data in enron_internal:
        for addr in _extract_email_addresses(email_data.get("from", "")):
            if _is_enron_email(addr):
                enron_senders.add(addr)
    
    for email_data in external:
        for addr in _extract_email_addresses(email_data.get("from", "")):
            if not _is_enron_email(addr):
                external_senders.add(addr)
    
    context.log.info(f"Unique Enron senders: {len(enron_senders)}")
    context.log.info(f"Unique external senders: {len(external_senders)}")
    
    return {
        "total_emails": total_count,
        "enron_email_count": len(enron_internal),
        "external_email_count": len(external),
        "unique_enron_senders": len(enron_senders),
        "unique_external_senders": len(external_senders),
        "enron_emails_path": str(enron_file),
        "external_emails_path": str(external_file),
    }

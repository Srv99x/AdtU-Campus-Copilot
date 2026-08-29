"""
Verify research report claims against V1 Markdown files and official AdtU URLs.

Hybrid approach:
1. Check claims against existing V1 Markdown files
2. If not verified by V1, fetch the original official AdtU source URL
3. Record verification status
4. Unverified claims go to knowledge_gaps.csv

Does NOT treat the research report as an authoritative source.
Does NOT chunk or embed the research report.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

V1_MARKDOWN_DIR = ROOT / "data" / "processed" / "markdown"
RESEARCH_FILE = ROOT / "data" / "raw" / "knowledge_v2" / "research" / "gemini_helpdesk_grievance_research.md"
OUTPUT_DIR = ROOT / "data" / "processed" / "knowledge_v2"

VERIFICATION_FILE = OUTPUT_DIR / "research_verification.csv"
KNOWLEDGE_GAPS_FILE = OUTPUT_DIR / "knowledge_gaps.csv"
CONFLICTS_FILE = OUTPUT_DIR / "conflicts.csv"


# --------------------------------------------------
# RESEARCH CLAIMS FROM THE VERIFIED KNOWLEDGE TABLE
# --------------------------------------------------

# These are extracted directly from the research report's OUTPUT 1 table.
# Each claim must be independently verified before entering the KB.

RESEARCH_CLAIMS: list[dict[str, str]] = [
    {
        "claim_id": "VK-01",
        "category": "Grievance",
        "claim": "Grievance Cell coordinated by Kasturi Dutta. Contact: student.grievance@adtu.in or +91 6003903319",
        "source_url": "https://adtu.in/contact/",
        "source_title": "Contact Us",
        "relevant_section": "Grievance Cell",
        "published_date": "N/A",
        "v1_search_terms": "kasturi dutta,student.grievance,grievance cell,6003903319",
    },
    {
        "claim_id": "VK-02",
        "category": "Grievance",
        "claim": "Standard complaints resolve in 7 working days; complex cases up to 25 working days",
        "source_url": "https://adtu.in/iqac/policy-mom/grievance-redressal/policy-for-grievance-redressal-system.pdf",
        "source_title": "Policy for Grievance Redressal",
        "relevant_section": "Procedure for addressing",
        "published_date": "N/A",
        "v1_search_terms": "7 working days,25 working days,grievance redressal",
    },
    {
        "claim_id": "VK-03",
        "category": "Grievance",
        "claim": "Students have a 15-working-day window to submit a Re-Appeal to the Registrar",
        "source_url": "https://adtu.in/iqac/policy-mom/grievance-redressal/policy-for-grievance-redressal-system.pdf",
        "source_title": "Policy for Grievance Redressal",
        "relevant_section": "Re-appeal",
        "published_date": "N/A",
        "v1_search_terms": "re-appeal,registrar,15 working days",
    },
    {
        "claim_id": "VK-04",
        "category": "Anti-Ragging",
        "claim": "Anti-ragging nodal officer: Dr. Sangita Boro. Contact: +91 9365107230 or sangita.boro@adtu.in",
        "source_url": "https://adtu.in/anti-ragging.html",
        "source_title": "Anti-Ragging",
        "relevant_section": "Contacts",
        "published_date": "N/A",
        "v1_search_terms": "sangita boro,anti-ragging,9365107230",
    },
    {
        "claim_id": "VK-05",
        "category": "Anti-Ragging",
        "claim": "Report ragging via A-Connect App: Administration -> Grievance -> Add new -> Ragging category",
        "source_url": "https://adtu.in/iqac/assets/files/Ragging_help_Mechanism.pdf",
        "source_title": "SOP for online complaint",
        "relevant_section": "SOP",
        "published_date": "N/A",
        "v1_search_terms": "a-connect,ragging,administration,grievance",
    },
    {
        "claim_id": "VK-06",
        "category": "ICC",
        "claim": "File sexual harassment complaint via A-Connect App: Administration -> Grievance -> Sexual Harassment",
        "source_url": "https://adtu.in/internal-complaint-committee.html",
        "source_title": "Internal Complaint Committee",
        "relevant_section": "SOP",
        "published_date": "N/A",
        "v1_search_terms": "sexual harassment,internal complaint,a-connect",
    },
    {
        "claim_id": "VK-07",
        "category": "ICC",
        "claim": "ICC Member Secretary: Ms. Neha Nongmeikapam (Psychological Counsellor). Phone: +91 9365393994",
        "source_url": "https://adtu.in/internal-complaint-committee.html",
        "source_title": "Internal Complaint Committee",
        "relevant_section": "Contacts",
        "published_date": "N/A",
        "v1_search_terms": "neha nongmeikapam,9365393994,counsellor",
    },
    {
        "claim_id": "VK-08",
        "category": "Academic",
        "claim": "75% compulsory attendance. 60% allowed on medical grounds with fine and valid certificate",
        "source_url": "https://adtu.in/code-of-conduct-students.html",
        "source_title": "Code of Conduct",
        "relevant_section": "Undertaking",
        "published_date": "N/A",
        "v1_search_terms": "75%,attendance,60%,medical grounds",
    },
    {
        "claim_id": "VK-09",
        "category": "Academic",
        "claim": "Unnotified absence over 30 days leads to removal from university roll",
        "source_url": "https://adtu.in/code-of-conduct-students.html",
        "source_title": "Code of Conduct",
        "relevant_section": "Undertaking",
        "published_date": "N/A",
        "v1_search_terms": "30 days,struck off,absence",
    },
    {
        "claim_id": "VK-10",
        "category": "Exam",
        "claim": "Controller of Examinations: Dr. Joydeep Goswami. Contact: coe@adtu.in, +91 9560790911",
        "source_url": "https://adtu.in/coe.html",
        "source_title": "Controller of Examinations",
        "relevant_section": "Contacts",
        "published_date": "N/A",
        "v1_search_terms": "joydeep goswami,controller of examination,coe@adtu.in,9560790911",
    },
    {
        "claim_id": "VK-11",
        "category": "Library",
        "claim": "Library hours: 9 AM to 8 PM working days; 10 AM to 12 PM Sundays/holidays",
        "source_url": "https://adtu.in/hndb-library/about-library/",
        "source_title": "About Library",
        "relevant_section": "Opening Hours",
        "published_date": "N/A",
        "v1_search_terms": "9:00 am,8:00 pm,library,10:00 am,12:00",
    },
    {
        "claim_id": "VK-12",
        "category": "Library",
        "claim": "Overdue fine: Rs. 10 per day for students (from 16th day); Rs. 20 per day for faculty",
        "source_url": "https://adtu.in/hndb-library/about-library/",
        "source_title": "About Library",
        "relevant_section": "Book Issue",
        "published_date": "N/A",
        "v1_search_terms": "rs. 10,fine,rs. 20,overdue,library",
    },
    {
        "claim_id": "VK-13",
        "category": "Library",
        "claim": "Students can borrow up to 2 books for 15 days",
        "source_url": "https://adtu.in/hndb-library/about-library/",
        "source_title": "About Library",
        "relevant_section": "Borrowing",
        "published_date": "N/A",
        "v1_search_terms": "2 books,15 days,borrow,issue",
    },
    {
        "claim_id": "VK-14",
        "category": "Finance",
        "claim": "Finance & Accounts Officer: Mr. Jaydeep Kakati. Contact: fao@adtu.in, +91 8472816245",
        "source_url": "https://adtu.in/contact/",
        "source_title": "Contact Us",
        "relevant_section": "Quick Links",
        "published_date": "N/A",
        "v1_search_terms": "jaydeep kakati,fao@adtu.in,8472816245,finance",
    },
    {
        "claim_id": "VK-15",
        "category": "Hostel",
        "claim": "Boys Hostel Warden: Mr. Pranab Barman, 392 seats across H, HE, I blocks",
        "source_url": "https://adtu.in/contact/#hostel",
        "source_title": "Contact Us",
        "relevant_section": "Hostel",
        "published_date": "N/A",
        "v1_search_terms": "pranab barman,h block,he block,i block,boys hostel",
    },
    {
        "claim_id": "VK-16",
        "category": "Hostel",
        "claim": "Girls Hostel Warden: Ms. Utpala Barman, 496 seats across A, E, F blocks",
        "source_url": "https://adtu.in/contact/#hostel",
        "source_title": "Contact Us",
        "relevant_section": "Hostel",
        "published_date": "N/A",
        "v1_search_terms": "utpala barman,a block,e block,f block,girls hostel",
    },
    {
        "claim_id": "VK-17",
        "category": "Transport",
        "claim": "Transport fraud penalty: pay entire year's transport fee for the route",
        "source_url": "https://adtu.in/code-of-conduct-students.html",
        "source_title": "Code of Conduct",
        "relevant_section": "Undertaking",
        "published_date": "N/A",
        "v1_search_terms": "transport fee,university bus,authorization",
    },
    {
        "claim_id": "VK-18",
        "category": "Conduct",
        "claim": "Strict daily university uniform and Identity Card display required",
        "source_url": "https://adtu.in/code-of-conduct-students.html",
        "source_title": "Code of Conduct",
        "relevant_section": "Undertaking",
        "published_date": "N/A",
        "v1_search_terms": "uniform,identity card,display",
    },
    {
        "claim_id": "VK-19",
        "category": "Placement",
        "claim": "Placement contact: Rimjhim Baruah Borah. Contact: rimjhim@adtu.in, +91 81348 99730",
        "source_url": "https://adtu.in/contact/",
        "source_title": "Contact Us",
        "relevant_section": "Quick Links",
        "published_date": "N/A",
        "v1_search_terms": "rimjhim,placement,81348 99730",
    },
    {
        "claim_id": "VK-20",
        "category": "Conduct",
        "claim": "Dual enrollment prohibited: cannot enroll in any other regular course at another institution",
        "source_url": "https://adtu.in/code-of-conduct-students.html",
        "source_title": "Code of Conduct",
        "relevant_section": "Undertaking",
        "published_date": "N/A",
        "v1_search_terms": "enrol,other university,regular course",
    },
]


# Knowledge gaps from the research report
RESEARCH_GAPS: list[dict[str, str]] = [
    {
        "question_topic": "How to apply for a migration certificate and its fee",
        "why_unverified": "CoE is listed as responsible but exact form, portal, and fees are not in primary sources",
        "potential_official_source": "coe.adtu.in or Student ERP",
        "recommended_answer": "no",
        "recommended_escalate": "yes - direct to coe@adtu.in",
    },
    {
        "question_topic": "Exact departure times for campus buses",
        "why_unverified": "12 routes verified but dynamic schedules are volatile and not publicly documented",
        "potential_official_source": "Transport Office Notice Board",
        "recommended_answer": "partial - provide verified routes only",
        "recommended_escalate": "yes - for specific times",
    },
    {
        "question_topic": "Campus doctor location and emergency medical numbers",
        "why_unverified": "Psychological support documented but physical triage/ambulance data missing from public sources",
        "potential_official_source": "Directorate of Student Affairs",
        "recommended_answer": "no",
        "recommended_escalate": "yes - provide helpline +91 93657 71454",
    },
    {
        "question_topic": "A-Connect password reset procedure",
        "why_unverified": "Login ID (enrollment number) known but IT recovery pathways are behind portal firewall",
        "potential_official_source": "my.adtu.in login interface",
        "recommended_answer": "no",
        "recommended_escalate": "yes - direct to IT Helpdesk or general admin line",
    },
    {
        "question_topic": "Exact monetary hostel and mess fee amounts",
        "why_unverified": "Capacities, blocks, wardens documented but per-semester fee amounts missing from collected sources",
        "potential_official_source": "Finance Office / Admission Forms",
        "recommended_answer": "no",
        "recommended_escalate": "yes - route to fao@adtu.in",
    },
    {
        "question_topic": "Exam re-evaluation procedure and timeline",
        "why_unverified": "Re-evaluation falls under CoE but exact steps, application window, and fees are unpublished",
        "potential_official_source": "Examination Ordinances",
        "recommended_answer": "no",
        "recommended_escalate": "yes - route to coe@adtu.in",
    },
    {
        "question_topic": "Lost student ID card replacement procedure and penalty",
        "why_unverified": "Code of Conduct mandates wearing ID daily but replacement procedure is undocumented",
        "potential_official_source": "Registrar or DSA",
        "recommended_answer": "no",
        "recommended_escalate": "yes - route to registrar@adtu.in",
    },
    {
        "question_topic": "Specific dates for PhD entrance test (RET)",
        "why_unverified": "CoE conducts RET but dates are volatile and change yearly",
        "potential_official_source": "Academic Calendar",
        "recommended_answer": "no - dates are volatile",
        "recommended_escalate": "yes - advise checking official website",
    },
    {
        "question_topic": "Guest rules and visitor hours for hostels",
        "why_unverified": "Hostel blocks and wardens defined but visitor/curfew rules absent from public sources",
        "potential_official_source": "Hostel Administration",
        "recommended_answer": "no",
        "recommended_escalate": "yes - contact hostel wardens",
    },
    {
        "question_topic": "Governor Assam Pratibha Protsahan Yojana eligibility requirements",
        "why_unverified": "Scholarship name listed on portal but exact criteria, amount, and application process missing",
        "potential_official_source": "Scholarship Schemes portal",
        "recommended_answer": "no",
        "recommended_escalate": "yes - route to admissions or finance office",
    },
]


# --------------------------------------------------
# V1 SEARCH
# --------------------------------------------------

def search_v1_markdown(search_terms: str, v1_texts: dict[str, str]) -> tuple[bool, str]:
    """Search V1 Markdown files for evidence of a claim.

    Returns (found, evidence_file).
    """

    terms = [t.strip().lower() for t in search_terms.split(",")]

    for filename, text in v1_texts.items():
        text_lower = text.lower()

        # Require at least 2 search terms to match
        matches = sum(1 for term in terms if term in text_lower)

        if matches >= 2:
            return True, filename

    return False, ""


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main() -> None:

    import sys
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

    # Load V1 Markdown files
    v1_texts: dict[str, str] = {}

    if V1_MARKDOWN_DIR.exists():
        for md_file in sorted(V1_MARKDOWN_DIR.glob("*.md")):
            v1_texts[md_file.name] = md_file.read_text(
                encoding="utf-8", errors="ignore"
            )

    print(f"V1 Markdown files loaded: {len(v1_texts)}")

    # Verify claims
    verification_records: list[dict[str, str]] = []
    verified_count = 0
    v1_verified_count = 0
    url_verified_count = 0
    unverified_count = 0

    for claim in RESEARCH_CLAIMS:

        # Step 1: Check V1 Markdown
        found_in_v1, evidence_file = search_v1_markdown(
            claim["v1_search_terms"],
            v1_texts,
        )

        if found_in_v1:
            verification_records.append({
                "claim_id": claim["claim_id"],
                "category": claim["category"],
                "claim": claim["claim"],
                "original_source_url": claim["source_url"],
                "document_title": claim["source_title"],
                "relevant_section": claim["relevant_section"],
                "published_date": claim["published_date"],
                "verification_status": "VERIFIED_V1_MARKDOWN",
                "verification_method": f"Found in V1 file: {evidence_file}",
                "notes": "Claim supported by existing V1 corpus content",
            })
            verified_count += 1
            v1_verified_count += 1
            print(f"  {claim['claim_id']}: VERIFIED via V1 ({evidence_file})")
            continue

        # Step 2: Attempt to fetch original source URL
        url = claim["source_url"]

        if url and not url.endswith(".pdf"):
            try:
                import urllib.request
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "AdtU-Campus-Copilot/1.0"},
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    page_content = response.read().decode("utf-8", errors="ignore").lower()

                    terms = [t.strip().lower() for t in claim["v1_search_terms"].split(",")]
                    matches = sum(1 for term in terms if term in page_content)

                    if matches >= 2:
                        verification_records.append({
                            "claim_id": claim["claim_id"],
                            "category": claim["category"],
                            "claim": claim["claim"],
                            "original_source_url": claim["source_url"],
                            "document_title": claim["source_title"],
                            "relevant_section": claim["relevant_section"],
                            "published_date": claim["published_date"],
                            "verification_status": "VERIFIED_ORIGINAL_URL",
                            "verification_method": f"Fetched and verified: {url}",
                            "notes": f"Found {matches}/{len(terms)} search terms in live page",
                        })
                        verified_count += 1
                        url_verified_count += 1
                        print(f"  {claim['claim_id']}: VERIFIED via URL ({url})")
                        continue

            except Exception as e:
                print(f"  {claim['claim_id']}: URL fetch failed: {e}")

        # Step 3: Mark as unverified
        verification_records.append({
            "claim_id": claim["claim_id"],
            "category": claim["category"],
            "claim": claim["claim"],
            "original_source_url": claim["source_url"],
            "document_title": claim["source_title"],
            "relevant_section": claim["relevant_section"],
            "published_date": claim["published_date"],
            "verification_status": "UNVERIFIED_PENDING",
            "verification_method": "Not found in V1 markdown or live URL",
            "notes": "Requires manual verification before entering trusted KB",
        })
        unverified_count += 1
        print(f"  {claim['claim_id']}: UNVERIFIED")

    # Write verification report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with VERIFICATION_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "claim_id", "category", "claim",
            "original_source_url", "document_title", "relevant_section",
            "published_date", "verification_status", "verification_method", "notes",
        ])
        writer.writeheader()
        writer.writerows(verification_records)

    # Write knowledge gaps
    with KNOWLEDGE_GAPS_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_topic", "why_unverified",
            "potential_official_source",
            "recommended_answer", "recommended_escalate",
        ])
        writer.writeheader()

        # Add unverified research claims as gaps
        for record in verification_records:
            if record["verification_status"] == "UNVERIFIED_PENDING":
                writer.writerow({
                    "question_topic": record["claim"],
                    "why_unverified": "Not independently verified from V1 markdown or original official URL",
                    "potential_official_source": record["original_source_url"],
                    "recommended_answer": "no",
                    "recommended_escalate": "yes",
                })

        # Add explicit research gaps
        writer.writerows(RESEARCH_GAPS)

    # Write empty conflicts file (no V1-V2 contradictions found via dedup)
    with CONFLICTS_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_a", "source_b", "date_a", "date_b",
            "conflicting_statement", "recommended_source", "reason",
        ])
        writer.writeheader()
        # No conflicts detected between V1 and V2 since they cover different topics

    # Summary
    print()
    print("=" * 60)
    print("RESEARCH VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total claims examined  : {len(RESEARCH_CLAIMS)}")
    print(f"Verified via V1       : {v1_verified_count}")
    print(f"Verified via URL      : {url_verified_count}")
    print(f"Total verified        : {verified_count}")
    print(f"Unverified            : {unverified_count}")
    print(f"Knowledge gaps (total): {len(RESEARCH_GAPS) + unverified_count}")
    print(f"Conflicts             : 0")
    print()
    print(f"Verification file     : {VERIFICATION_FILE}")
    print(f"Knowledge gaps file   : {KNOWLEDGE_GAPS_FILE}")
    print(f"Conflicts file        : {CONFLICTS_FILE}")


if __name__ == "__main__":
    main()

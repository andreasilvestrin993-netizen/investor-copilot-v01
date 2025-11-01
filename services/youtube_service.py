"""
YouTube service for channel discovery and transcript fetching
"""
import requests
import re
from datetime import date
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

def is_video_url(url: str) -> bool:
    """Check if URL is a YouTube video"""
    return ("youtube.com/watch" in url) or ("youtu.be/" in url)

def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL"""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def channel_rss_url(channel_id: str) -> str:
    """Get RSS feed URL for a YouTube channel"""
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

def resolve_channel_id(channel_url_or_handle: str) -> str | None:
    """
    Resolve YouTube channel URL or handle to channel ID.
    Returns channel ID (UC...) or None if not found.
    """
    try:
        if channel_url_or_handle.startswith("UC"):
            return channel_url_or_handle
        url = channel_url_or_handle.strip()
        if not url.startswith("http"):
            url = f"https://www.youtube.com/{url.lstrip('/')}"
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        m = re.search(r'"channelId":"(UC[0-9A-Za-z_\-]{20,})"', r.text)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None

def fetch_latest_from_channel(channel_id: str, since_date: date) -> list[dict]:
    """
    Fetch latest videos from a YouTube channel.
    Returns videos from the last 7 days, or latest 3 if none found.
    """
    import xml.etree.ElementTree as ET
    url = channel_rss_url(channel_id)
    vids = []
    all_vids = []
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text
            link = entry.find('atom:link', ns).attrib.get('href')
            published = entry.find('atom:published', ns).text
            pub_date = date.fromisoformat(published[:10])
            video_data = {'title': title, 'url': link, 'published': published}
            all_vids.append(video_data)
            
            # Get videos from the last 7 days
            days_diff = (since_date - pub_date).days
            if 0 <= days_diff <= 7:
                vids.append(video_data)
        
        # If no recent videos found, get the latest 3 videos as baseline
        if not vids and all_vids:
            vids = all_vids[:3]  # RSS feeds are typically ordered by date descending
            
    except Exception:
        pass
    return vids

def fetch_transcript_text(video_id: str) -> str | None:
    """
    Fetch English transcript for a YouTube video.
    Returns transcript text or None if unavailable.
    """
    try:
        parts = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([p["text"] for p in parts if p.get("text")])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception:
        return None

import os
from typing import Any, Dict, List
import streamlit as st

# High-availability public STUN servers for standard NAT discovery
DEFAULT_STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
    "stun:stun2.l.google.com:19302",
    "stun:stun3.l.google.com:19302",
    "stun:stun4.l.google.com:19302",
    "stun:stun.cloudflare.com:3478",
    "stun:openrelay.metered.ca:80",
]

# Free public community TURN relay from Open Relay Project (fallback when symmetric NAT blocks direct P2P)
DEFAULT_OPENRELAY_TURN = {
    "urls": [
        "turn:openrelay.metered.ca:80",
        "turn:openrelay.metered.ca:443",
        "turn:openrelay.metered.ca:443?transport=tcp",
    ],
    "username": "openrelayproject",
    "credential": "openrelayproject",
}


def get_ice_servers() -> List[Dict[str, Any]]:
    """
    Constructs the ICE servers configuration for WebRTC.
    Checks:
    1. Dedicated TURN configuration from st.secrets['webrtc'] if provided.
    2. Environment variables (TURN_URL, TURN_USERNAME, TURN_CREDENTIAL) if set.
    3. Resilient public STUN servers + community TURN fallback for cloud hosting.
    """
    ice_servers: List[Dict[str, Any]] = [
        {"urls": DEFAULT_STUN_SERVERS}
    ]

    has_custom_turn = False

    # 1. Check Streamlit Secrets: st.secrets["webrtc"]
    try:
        if hasattr(st, "secrets") and "webrtc" in st.secrets:
            webrtc_sec = st.secrets["webrtc"]
            if isinstance(webrtc_sec, dict):
                # Check for explicit list
                if "ice_servers" in webrtc_sec and isinstance(webrtc_sec["ice_servers"], list):
                    ice_servers = webrtc_sec["ice_servers"]
                    return ice_servers

                turn_url = webrtc_sec.get("turn_url")
                turn_user = webrtc_sec.get("turn_username")
                turn_cred = webrtc_sec.get("turn_credential") or webrtc_sec.get("turn_password")

                if turn_url:
                    urls = turn_url if isinstance(turn_url, list) else [str(turn_url)]
                    custom_turn: Dict[str, Any] = {"urls": urls}
                    if turn_user:
                        custom_turn["username"] = str(turn_user)
                    if turn_cred:
                        custom_turn["credential"] = str(turn_cred)
                    ice_servers.append(custom_turn)
                    has_custom_turn = True
    except Exception:
        pass

    # 2. Check Environment Variables fallback
    if not has_custom_turn:
        env_turn_url = os.environ.get("TURN_URL") or os.environ.get("WEBRTC_TURN_URL")
        if env_turn_url:
            env_user = os.environ.get("TURN_USERNAME") or os.environ.get("WEBRTC_TURN_USERNAME")
            env_cred = os.environ.get("TURN_CREDENTIAL") or os.environ.get("TURN_PASSWORD") or os.environ.get("WEBRTC_TURN_PASSWORD")
            urls = [u.strip() for u in env_turn_url.split(",") if u.strip()]
            custom_turn = {"urls": urls}
            if env_user:
                custom_turn["username"] = env_user
            if env_cred:
                custom_turn["credential"] = env_cred
            ice_servers.append(custom_turn)
            has_custom_turn = True

    # 3. Add community OpenRelay TURN if no dedicated TURN server was configured
    if not has_custom_turn:
        ice_servers.append(DEFAULT_OPENRELAY_TURN)

    return ice_servers


def get_rtc_configuration() -> Dict[str, Any]:
    """Returns the dictionary format expected by streamlit-webrtc."""
    return {"iceServers": get_ice_servers()}


# Recommended WebRTC camera constraints for reliable streaming on cloud containers
DEFAULT_MEDIA_CONSTRAINTS = {
    "video": {
        "width": {"ideal": 640, "max": 1280},
        "height": {"ideal": 480, "max": 720},
        "frameRate": {"ideal": 30, "max": 30},
    },
    "audio": False,
}

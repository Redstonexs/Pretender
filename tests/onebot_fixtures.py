"""Captured-style OneBot v11 payloads for the adapter/media tests.

These mirror real OneBot v11 / NapCat reverse-WebSocket frames (message_format
= array): group/private messages, at/reply/face/sticker/image segments, self
echoes (``message_sent``), meta heartbeats/lifecycle, notices, requests, and
API action responses for echo correlation.
"""

from __future__ import annotations

# ── inbound message events ──────────────────────────────────────────────────

GROUP_TEXT_IMAGE = {
    "time": 1700000001,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "group",
    "sub_type": "normal",
    "message_id": 12345,
    "group_id": 111111,
    "user_id": 222222,
    "message": [
        {"type": "text", "data": {"text": "看看这个 "}},
        {
            "type": "image",
            "data": {
                "file": "abc.jpg",
                "url": "https://example.com/abc.jpg",
            },
        },
    ],
    "raw_message": "看看这个 [图片]",
    "font": 14,
    "sender": {
        "user_id": 222222,
        "nickname": "小明",
        "card": "小明",
        "role": "member",
    },
}

PRIVATE_TEXT = {
    "time": 1700000002,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "private",
    "sub_type": "friend",
    "message_id": 12346,
    "user_id": 333333,
    "message": [{"type": "text", "data": {"text": "在吗"}}],
    "raw_message": "在吗",
    "sender": {"user_id": 333333, "nickname": "小红"},
}

GROUP_AT = {
    "time": 1700000003,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "group",
    "message_id": 12347,
    "group_id": 111111,
    "user_id": 444444,
    "message": [
        {"type": "at", "data": {"qq": "10001", "name": "麦麦"}},
        {"type": "text", "data": {"text": " 你好"}},
    ],
    "raw_message": "[CQ:at,qq=10001] 你好",
    "sender": {"user_id": 444444, "nickname": "阿强"},
}

GROUP_REPLY = {
    "time": 1700000004,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "group",
    "message_id": 12348,
    "group_id": 111111,
    "user_id": 555555,
    "message": [
        {"type": "reply", "data": {"id": "12345"}},
        {"type": "text", "data": {"text": "同意"}},
    ],
    "raw_message": "同意",
    "sender": {"user_id": 555555, "nickname": "小李"},
}

GROUP_FACE = {
    "time": 1700000005,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "group",
    "message_id": 12349,
    "group_id": 111111,
    "user_id": 666666,
    "message": [
        {"type": "face", "data": {"id": "178"}},
        {"type": "text", "data": {"text": "哈哈"}},
    ],
    "raw_message": "[CQ:face,id=178]哈哈",
    "sender": {"user_id": 666666, "nickname": "小刚"},
}

GROUP_STICKER = {
    "time": 1700000006,
    "self_id": 10001,
    "post_type": "message",
    "message_type": "group",
    "message_id": 12350,
    "group_id": 111111,
    "user_id": 777777,
    "message": [
        {
            "type": "sticker",
            "data": {"file": "sticker.webp", "url": "https://example.com/sticker.webp"},
        },
    ],
    "raw_message": "[贴纸]",
    "sender": {"user_id": 777777, "nickname": "小美"},
}

# ── self echoes (message_sent) ──────────────────────────────────────────────

SELF_ECHO_GROUP = {
    "time": 1700000010,
    "self_id": 10001,
    "post_type": "message_sent",
    "message_type": "group",
    "message_id": 90001,
    "group_id": 111111,
    "user_id": 10001,
    "message": [{"type": "text", "data": {"text": "大家好"}}],
    "raw_message": "大家好",
    "sender": {"user_id": 10001, "nickname": "麦麦"},
}

SELF_ECHO_PRIVATE = {
    "time": 1700000011,
    "self_id": 10001,
    "post_type": "message_sent",
    "message_type": "private",
    "message_id": 90002,
    "user_id": 10001,
    "target_id": 333333,
    "message": [{"type": "text", "data": {"text": "在的"}}],
    "raw_message": "在的",
    "sender": {"user_id": 10001, "nickname": "麦麦"},
}

# ── meta events ─────────────────────────────────────────────────────────────

HEARTBEAT = {
    "time": 1700000020,
    "self_id": 10001,
    "post_type": "meta_event",
    "meta_event_type": "heartbeat",
    "status": {"online": True, "good": True},
    "interval": 3000,
}

LIFECYCLE = {
    "time": 1700000000,
    "self_id": 10001,
    "post_type": "meta_event",
    "meta_event_type": "lifecycle",
    "sub_type": "connect",
}

# ── notice / request ────────────────────────────────────────────────────────

NOTICE_POKE = {
    "time": 1700000030,
    "self_id": 10001,
    "post_type": "notice",
    "notice_type": "notify",
    "sub_type": "poke",
    "group_id": 111111,
    "user_id": 222222,
    "target_id": 10001,
}

REQUEST_FRIEND = {
    "time": 1700000040,
    "self_id": 10001,
    "post_type": "request",
    "request_type": "friend",
    "user_id": 888888,
    "comment": "",
    "flag": "abc123",
}

# ── API action responses (echo correlation) ─────────────────────────────────

def api_ok(echo: str, message_id: int) -> dict:
    return {"status": "ok", "retcode": 0, "data": {"message_id": message_id}, "echo": echo}


def api_fail(echo: str, retcode: int = -1, message: str = "bad request") -> dict:
    return {
        "status": "failed",
        "retcode": retcode,
        "data": None,
        "echo": echo,
        "message": message,
    }


FIXTURES = {
    "group_text_image": GROUP_TEXT_IMAGE,
    "private_text": PRIVATE_TEXT,
    "group_at": GROUP_AT,
    "group_reply": GROUP_REPLY,
    "group_face": GROUP_FACE,
    "group_sticker": GROUP_STICKER,
    "self_echo_group": SELF_ECHO_GROUP,
    "self_echo_private": SELF_ECHO_PRIVATE,
    "heartbeat": HEARTBEAT,
    "lifecycle": LIFECYCLE,
    "notice_poke": NOTICE_POKE,
    "request_friend": REQUEST_FRIEND,
}

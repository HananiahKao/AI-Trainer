# views.py — Django view for Agent TARS (DeepSeek v3) + Granite 3.2-vision tools
import json, uuid, time, requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

OLLAMA_MODEL    = "granite3.2-vision"           # correct model name
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

@csrf_exempt
@require_POST
def chat_view(request):
    # 1) Parse JSON
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse({"error": f"Invalid JSON: {e}"}, status=400)

    # 2) Validate messages
    messages = data.get("messages")
    if not isinstance(messages, list):
        return JsonResponse({
            "error": f'Invalid "messages": expected list, got {type(messages).__name__}'
        }, status=400)
    sys_messages_from_me = {
    "role": "system",
    "content": "Don't attempt to execute tools by writing its name in your response."
}
    messages.insert(0, sys_messages_from_me)
    # 3) Build payload for Ollama (Granite 3.2 supports tools)  [oai_citation:3‡Ollama](https://ollama.com/library/granite3.2?utm_source=chatgpt.com)

    payload = {
        "model":   OLLAMA_MODEL,
        "messages": messages,
        "stream":   False
    }
    # Agent TARS expects "tools" list
    tools = data.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            return JsonResponse({"error": f'"tools" must be list, got {type(tools).__name__}'}, status=400)
        payload["tools"] = tools

    # tool_choice (DeepSeek v3 uses "tool_choice")  [oai_citation:4‡IBM - United States](https://www.ibm.com/think/tutorials/local-tool-calling-ollama-granite?utm_source=chatgpt.com)
    if "tool_choice" in data:
        payload["tool_choice"] = data["tool_choice"]
    # 4) Call Ollama
    try:
        print(prettify_and_colorize_JSON(payload))
        resp = requests.post(OLLAMA_ENDPOINT, json=payload)
        resp.raise_for_status()
    except requests.RequestException as e:
        return JsonResponse({"error": f"Failed to contact model API: {e}"}, status=500)

    # 5) Parse response JSON
    try:
        ollama_data = resp.json()
    except ValueError as e:
        return JsonResponse({"error": f"Invalid JSON from model API: {e}"}, status=500)

    msg = ollama_data.get("message")
    if not isinstance(msg, dict):
        return JsonResponse({"error": f"Unexpected response format: {msg}"}, status=500)

    # 6) If model invoked a tool, return DeepSeek-style tool_calls  [oai_citation:5‡IBM - United States](https://www.ibm.com/granite/docs/models/granite/?utm_source=chatgpt.com)
    print("\n\nRespond:\n\n")
    print(prettify_and_colorize_JSON(ollama_data))

    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        tc = tool_calls[0]
        fn = tc.get("function", {})



        print(prettify_and_colorize_JSON({
            "id":       "chatcmpl-" + uuid.uuid4().hex,
            "object":   "chat.completion",
            "created":  int(time.time()),
            "model":    OLLAMA_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role":    "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id":   tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name":      fn.get("name"),
                                    "arguments": json.dumps(fn.get("arguments", {}))
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        }))



        return JsonResponse({
            "id":       "chatcmpl-" + uuid.uuid4().hex,
            "object":   "chat.completion",
            "created":  int(time.time()),
            "model":    OLLAMA_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role":    "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id":   tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name":      fn.get("name"),
                                    "arguments": json.dumps(fn.get("arguments", {}))
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        })

    # 7) Otherwise, return normal assistant reply
    print(prettify_and_colorize_JSON({
        "id":       "chatcmpl-" + uuid.uuid4().hex,
        "object":   "chat.completion",
        "created":  int(time.time()),
        "model":    OLLAMA_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role":    "assistant",
                    "content": msg.get("content", "")
                },
                "finish_reason": "stop"
            }
        ]
    }))
    return JsonResponse({
        "id":       "chatcmpl-" + uuid.uuid4().hex,
        "object":   "chat.completion",
        "created":  int(time.time()),
        "model":    OLLAMA_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role":    "assistant",
                    "content": msg.get("content", "")
                },
                "finish_reason": "stop"
            }
        ]
    })


def prettify_and_colorize_JSON(data):
    prittied = json.dumps(data, indent=2)
    from pygments import highlight, lexers, formatters
    colorful_json = highlight(prittied, lexers.JsonLexer(), formatters.Terminal256Formatter())
    return colorful_json


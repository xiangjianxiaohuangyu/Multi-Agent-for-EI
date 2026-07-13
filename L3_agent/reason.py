"""L3 推理模块（Reason）。

负责调本地 Ollama（OpenAI 兼容入口）/ 其它 OpenAI 兼容服务，
把感知快照作为一条 ``user`` 消息发到 ``/v1/chat/completions``，解析回复。

本模块对外暴露两个语义不同的入口方法，**职责分离**：

1. ``recognize_intent_and_plan(snapshot)`` —— 任务识别 + 初步规划
   （先验：飞行节点想做什么；输出：候选动作集合）

2. ``reasoning_and_decision(snapshot)``  —— 推理 + 决策
   （综合感知做最终动作裁决；输出：单一最终动作 + 理由）

两个方法共用一条 OpenAI 格式的 HTTP 通道（``_post_chat`` 私有方法），
但**使用不同提示词**来约束模型走不同思路。

调用约定：

    >>> r = Reason()
    >>> r.recognize_intent_and_plan(snapshot)
    {'status': 'ok',  'text': '...', 'model': 'qwen2:7b'}
    >>> r.reasoning_and_decision(snapshot)
    {'status': 'ok',  'text': '...', 'model': 'qwen2:7b'}

错误一律返回 ``status='error'``，**不抛异常**，由 Agent 按状态分流。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict


__all__ = ["Reason"]


OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class Reason:
    """调用本地 OpenAI 兼容端点（默认 Ollama）的推理客户端。"""

    # 默认指向本地 Ollama；改成官方 OpenAI / LM Studio / vLLM 等时只换 url / model 即可。
    DEFAULT_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "qwen2:7b"
    DEFAULT_TIMEOUT: float = 30.0

    # 两个方法各自的提示词：不同任务走不同思路。
    SYSTEM_PROMPT_INTENT_AND_PLAN: str = (
        "你是一名 FANET（Flying Ad-Hoc Network）任务识别与初步规划助手。"
        "你的职责是：在**不立即下决定**的前提下，"
        "基于输入的感知快照识别当前飞行节点的可能意图，"
        "并给出若干候选动作（每条一行，简短）。"
        "只输出识别结果和候选动作，不要输出最终决策。"
        "回复用中文。"
    )
    SYSTEM_PROMPT_REASONING_AND_DECISION: str = (
        "你是一名 FANET 推理与最终决策助手。"
        "你的职责是：基于输入的感知快照做综合推理，"
        "输出**唯一一个最终动作**及其简短理由（中文）。"
        "动作名必须是下列之一：hold / reroute / low_battery / rtb / continue，"
        "其它动作不允许。回复只输出 1 行：<action>，<reason>。"
    )

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._url: str = url.rstrip("/")
        self._model: str = model
        self._timeout: float = float(timeout)

    # ------------------------------------------------------------------ 配置访问

    def get_url(self) -> str:
        return self._url

    def get_model(self) -> str:
        return self._model

    # ------------------------------------------------------------------ 业务入口：两个不同的推理任务

    def recognize_intent_and_plan(
        self,
        perception_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """任务识别 + 初步规划。

        提示词偏"识别"风格：让模型先回答"想做什么 / 该考虑哪些候选动作"，
        而不是直接给最终动作。

        Parameters
        ----------
        perception_snapshot : dict
            L1 ``Perception.get()`` 返回的完整感知快照。

        Returns
        -------
        dict
            ``{"status": "ok"|"error", "text"|"error": "...", "model": "..."}``
        """
        user_content = self._build_user_content(
            system_prompt=self.SYSTEM_PROMPT_INTENT_AND_PLAN,
            perception_snapshot=perception_snapshot,
            task_label="意图识别 + 候选动作规划",
        )
        return self._post_chat(user_content)

    def reasoning_and_decision(
        self,
        perception_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """综合推理 + 最终决策。

        提示词偏"决策"风格：约束模型只能从固定动作集合里挑一个并给理由。

        Parameters
        ----------
        perception_snapshot : dict
            L1 ``Perception.get()`` 返回的完整感知快照。

        Returns
        -------
        dict
            ``{"status": "ok"|"error", "text"|"error": "...", "model": "..."}``
        """
        user_content = self._build_user_content(
            system_prompt=self.SYSTEM_PROMPT_REASONING_AND_DECISION,
            perception_snapshot=perception_snapshot,
            task_label="综合推理 + 最终决策",
        )
        return self._post_chat(user_content)

    # ------------------------------------------------------------------ 内部辅助

    def _build_user_content(
        self,
        system_prompt: str,
        perception_snapshot: Dict[str, Any],
        task_label: str,
    ) -> str:
        """拼成一条 user message：任务标签 + 简要 system 约束 + 快照。

        这里走"全部塞进 user.content"的妥协写法 —— 部分 OpenAI 兼容端点
        对 ``role=system`` 处理不一致，统一只走 user 最稳。
        """
        snapshot_json = json.dumps(perception_snapshot, ensure_ascii=False, default=str)
        return (
            f"[任务]\n{task_label}\n\n"
            f"[系统约束]\n{system_prompt}\n\n"
            f"[感知快照]\n{snapshot_json}"
        )

    def _post_chat(self, user_content: str) -> Dict[str, Any]:
        """把一条 user 消息 POST 到 ``/v1/chat/completions``，解析回复。

        任何 HTTP / 网络 / JSON 异常都被兜底为 ``status=error``，不抛。
        """
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url=f"{self._url}{OPENAI_CHAT_COMPLETIONS_PATH}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # OpenAI 标准路径；少数兼容实现会返回 ``choices[0].text``，做兜底
            choices = payload.get("choices") or []
            text = ""
            if choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content") or choices[0].get("text", "") or ""
            return {"status": "ok", "text": text, "model": self._model}
        except urllib.error.HTTPError as exc:
            return {
                "status": "error",
                "error": f"HTTPError {exc.code}: {exc.reason}",
                "model": self._model,
            }
        except urllib.error.URLError as exc:
            return {
                "status": "error",
                "error": f"URLError: {exc.reason} (is the OpenAI-compatible server running at {self._url}?)",
                "model": self._model,
            }
        except Exception as exc:  # noqa: BLE001 - 调用方按 status 处理，不抛
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model": self._model,
            }

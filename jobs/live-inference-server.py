import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple


READY_MARKER = "PEACE_EVENT: LIVE_INFERENCE_READY"
REQUEST_MARKER = "PEACE_EVENT: LIVE_INFERENCE_REQUEST"


class LiveInferenceService:
    def __init__(self, model_name: str, device: str, max_length: int, cache_dir: str) -> None:
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "hub"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "transformers"), exist_ok=True)
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cache_dir, "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))

        logging.info("Importing torch and transformers...")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logging.info("Post imports...")
        self.torch = torch
        self.model_name = model_name
        self.cache_dir = cache_dir
        if device == "auto":
            logging.info("Auto-detecting device with torch.cuda.is_available()...")
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected_device = device
        logging.info("Selected device=%s", selected_device)
        self.device = torch.device(selected_device)
        self.max_length = max_length

        load_start = time.time()
        logging.info(
            "Starting live inference service with model=%s device=%s cache_dir=%s",
            model_name,
            self.device,
            self.cache_dir,
        )
        logging.info("Loading tokenizer for %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        logging.info("Tokenizer loaded.")
        logging.info("Loading model %s...", model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=5,
            cache_dir=self.cache_dir,
        )
        logging.info("Model loaded. Moving model to %s...", self.device)
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            self.torch.cuda.synchronize()
        logging.info("[TIMER] live_inference_model_load_time: %.4f seconds", time.time() - load_start)
        logging.info("%s model=%s device=%s", READY_MARKER, model_name, self.device)

    def infer(self, text: str) -> Dict[str, object]:
        start = time.time()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = int(self.torch.argmax(logits, dim=-1).item())
            scores = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()[0]

        if self.device.type == "cuda":
            self.torch.cuda.synchronize()

        latency_ms = (time.time() - start) * 1000
        logging.info("%s latency_ms=%.2f prediction=%s", REQUEST_MARKER, latency_ms, prediction)
        return {
            "model": self.model_name,
            "prediction": prediction,
            "scores": scores,
            "latency_ms": latency_ms,
        }


def make_handler(service: LiveInferenceService):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Tuple[Dict[str, object], bool]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}, False
            raw_body = self.rfile.read(content_length)
            try:
                return json.loads(raw_body.decode("utf-8")), True
            except json.JSONDecodeError:
                return {}, False

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "model": service.model_name})
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/infer":
                self._send_json(404, {"error": "not found"})
                return

            payload, ok = self._read_json()
            if not ok:
                self._send_json(400, {"error": "invalid json body"})
                return

            text = str(payload.get("text", "")).strip()
            if not text:
                self._send_json(400, {"error": "missing non-empty 'text'"})
                return

            try:
                self._send_json(200, service.infer(text))
            except Exception as exc:
                logging.exception("Inference request failed.")
                self._send_json(500, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            logging.info("HTTP: " + format, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Live PEACE inference HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_name", default="bert-large-cased")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(os.path.expanduser("~"), ".cache", "peace-hf"),
        help="Writable local cache directory used for Hugging Face model/tokenizer downloads.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - LIVE-INFER - %(message)s", force=True)
    service = LiveInferenceService(args.model_name, args.device, args.max_length, args.cache_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    logging.info("Serving live inference on %s:%s", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()

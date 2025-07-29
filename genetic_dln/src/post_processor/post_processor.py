import json
import os

from genetic_dln.src.constants import constants
from genetic_dln.src.custom_logger.custom_logger import CustomLogger


class PostProcessor:
    def __init__(self):
        log_dir = os.path.join(constants.root_folder, "logs")
        self.logger = CustomLogger(log_dir=log_dir, logger_name="PostProcessor").logger

    def extract_json_objects(self, s, start_char, end_char):
        self.logger.info("Extracting JSON objects from response text.")
        json_objects = []
        brace_stack = []
        start_idx = None

        for idx, char in enumerate(s):
            if char == start_char:
                brace_stack.append(start_char)
                if start_idx is None:
                    start_idx = idx
            elif char == end_char:
                brace_stack.pop()
                if not brace_stack:
                    json_str = s[start_idx:idx + 1]
                    try:
                        json_obj = json.loads(json_str)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError as e:
                        self.logger.warning("JSON decoding failed: %s \n String: %s", e, json_str)
                        return json_objects

        self.logger.info("Successfully extracted %d JSON object(s)", len(json_objects))
        return json_objects
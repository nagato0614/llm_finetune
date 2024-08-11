#! /bin/bash

./llama.cpp/llama-cli -m ./model/unsloth.Q4_K_M.gguf  -n 4096 --repeat_penalty 1.0 --color -i -r "User:" -f ./llama.cpp/prompts/chat-with-bob.txt
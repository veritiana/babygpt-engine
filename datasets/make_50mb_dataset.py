import os
import random

SOURCE_FILE = "sk.txt"
OUTPUT_FILE = "dataset_10MB.txt"

TARGET_SIZE = 10 * 1024 * 1024  # 10 MB
BUFFER_SIZE = 100000


def generate():

    out = open(OUTPUT_FILE, "w", encoding="utf-8")

    written = 0
    buffer = []

    with open(SOURCE_FILE, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            buffer.append(line)

            if len(buffer) >= BUFFER_SIZE:

                random.shuffle(buffer)

                for l in buffer:

                    size = len(l.encode("utf-8"))

                    if written + size > TARGET_SIZE:
                        out.close()
                        print("DONE:", OUTPUT_FILE)
                        print("SIZE:", written)
                        return

                    out.write(l)
                    written += size

                buffer.clear()

    out.close()


if __name__ == "__main__":
    generate()
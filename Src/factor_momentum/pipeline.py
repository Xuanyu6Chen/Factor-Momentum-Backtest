from .data_fetch import main as fetch_main
from .data_clean import main as clean_main
from .returns import main as returns_main

def main():
    fetch_main()
    clean_main()
    returns_main()
    print("Pipeline complete ✅")

if __name__ == "__main__":
    main()
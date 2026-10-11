"""Biweight benchmark (kept for compatibility): same as ``bench.py --method biweight``"""
import sys
import bench

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--method", "biweight"] + sys.argv[1:]
    bench.main()

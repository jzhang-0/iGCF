kernprof -v -l src/script/main.py -m "Random" --datan "ml-100k" -p 0.001 --online_iter 25 -E "UCB" --cuda 2 --K 1 -d 64 --online_rec_total_num 40
python -m line_profiler main.py.lprof > results2.txt

tmux kill-session -t 0
cd /home/admin/Workspace

((total=2090))
((index=0))
((succ=0))
((fail=0))

keys_file=entrance.txt
pages_dir=data

for key in $(cat $keys_file); do
    ((index++))

    if [ -f $pages_dir/$key.html ]; then
        ((succ++))
        continue
    fi

    if [ -f $pages_dir/$key.fail.html ]; then
        ((fail++))
        continue
    fi

    date +"%Y.%m.%d %H:%M:%S key=$key"

    # 不建议加 --silent, 用于观察网络连接情况.
    curl -x 172.27.127.72:8888 \
        -H "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36" \
        -H "referer: http://www.kaom.net/si_x8.php?c=$key" \
        -X POST -d "city=$key" http://www.kaom.net/si_x8s.php > $pages_dir/temp.html

    if grep '提示：格內數字為聲調、左右滾動條在下方' $pages_dir/temp.html > /dev/null; then
        ((succ++))
        mv $pages_dir/temp.html $pages_dir/$key.html
    else
        ((fail++))
        mv $pages_dir/temp.html $pages_dir/$key.fail.html
    fi
    echo total=$((total)) index=$((index)) succ=$((succ)) fail=$((fail))
    echo

    if [ -f $pages_dir/$key.html ]; then
        sleep $((RANDOM % 60 + 120))
    fi

    if [ -f $pages_dir/$key.fail.html ]; then
        echo failed download $pages_dir/$key.fail.html
        # sleep $((60*60*24))
    fi
done

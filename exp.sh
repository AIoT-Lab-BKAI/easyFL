for i in $(find "scripts" -name \*.sh | sort -d); do # Not recommended, will break on whitespace
    echo "$i"
    bash "$i"
done

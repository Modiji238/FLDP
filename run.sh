python server.py &

sleep 3

for i in 0 1 2 3 4
do
  python client.py $i &
done
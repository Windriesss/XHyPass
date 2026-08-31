brctl addbr xenbr0
ifconfig xenbr0 202.197.68.50
ifconfig eth0 0.0.0.0
brctl addif xenbr0 eth0

echo 1 > /proc/sys/net/ipv4/ip_forward
iptables -F
iptables -t nat -F
iptables -A FORWARD -i xenbr0 -o eth0 -j ACCEPT
iptables -A FORWARD -i eth0 -o xenbr0 -m state --state RELATED,ESTABLISHED -j ACCEPT


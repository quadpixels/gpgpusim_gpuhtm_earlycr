all: Cartographer.so

Cartographer.so:
	g++ -g -O3 -fPIC -c Cartographer.cc -I../src -std=c++11
	g++ -g -O3 -shared -o Cartographer.so Cartographer.o -lsqlite3
	
clean:
	[ -f Cartographer.o ] && rm -v Cartographer.o
	[ -f Cartographer.so ] && rm -v Cartographer.so

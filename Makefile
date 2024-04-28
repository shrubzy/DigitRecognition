install:
	test -d env || python3.10 -m venv env
	. env/bin/activate; pip3 install -Ur requirements.txt
	
clean:
	rm -rf env
	find -iname "*.pyc" -delete

UFgui.jar:  UFgui.java UFhelp.html
	javac UFgui.java
	jar cfe UFgui.jar UFgui *.class UFhelp.html
	- $(RM) *.class

run: UFgui.jar
	java -jar UFgui.jar

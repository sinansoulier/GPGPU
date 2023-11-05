# Définir le compilateur
CXX=g++

# Définir les flags de compilation
CXXFLAGS=-Wall `pkg-config --cflags opencv4`

# Définir les bibliothèques à lier
LIBS=`pkg-config --libs opencv4`

# Nom de l'exécutable à générer
TARGET=main

# Liste de fichiers objet que le projet requiert
OBJS=lab_space.o remove_noise.o tools.o main.o

# La première règle est celle exécutée lorsque seulement "make" est tapé
$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS) $(LIBS)

# Règles pour créer les fichiers objet
lab_space.o: lab_space.cpp lab_space.hh
	$(CXX) $(CXXFLAGS) -c lab_space.cpp

remove_noise.o: remove_noise.cpp remove_noise.hh
	$(CXX) $(CXXFLAGS) -c remove_noise.cpp

tools.o: tools.cpp tools.hh
	$(CXX) $(CXXFLAGS) -c tools.cpp

main.o: main.cpp lab_space.hh remove_noise.hh tools.hh
	$(CXX) $(CXXFLAGS) -c main.cpp

# Règle pour nettoyer les fichiers compilés (uniquement les fichiers intermédiaires)
clean:
	rm -f $(OBJS) $(TARGET)

# Règle pour nettoyer tout ce qui peut être généré
distclean: clean
	rm -f *.jpg

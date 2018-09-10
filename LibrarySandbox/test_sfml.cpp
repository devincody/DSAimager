#include <iostream>
#include <SFML/Graphics.hpp>

int main(){

        std::cout << "hei" << std::endl;

        sf::Uint8 *img = new sf::Uint8[100*100*4];
        sf::Image image;
        image.create(100,100, img);

        for (int i = 0; i < 100*100*2; i++){
                img[i] = 200;
        }

        image.saveToFile("testpic.jpg");

        delete[] img;
        return 0;
}


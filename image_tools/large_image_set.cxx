#include <cmath>

#include <fstream>
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::ios;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "large_image_set.hxx"

#include "stdint.h"

#ifdef _HAS_TIFF_
#include <sstream>
using std::ostringstream;

#include "tiff.h"
#include "tiffio.h"
#endif

LargeImage::LargeImage(ifstream &infile, int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const LargeImages *_images) {
    number_subimages = _number_subimages;
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    images = _images;

    //cout << "channels: " << channels << ", height: " << height << ", width: " << width << endl;

    pixels = vector< vector< vector<uint8_t> > >(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

    char* c_pixels = new char[channels * width * height];

    infile.read( c_pixels, sizeof(char) * channels * width * height);

    int current = 0;
    for (int z = 0; z < channels; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[z][y][x] = (uint8_t)c_pixels[current];
                current++;
                //cout << "pixels[" << z << "][" << y << "][" << x << "]: " << (int)pixels[z][y][x] << endl;
            }
        }
    }
    delete [] c_pixels;
}

LargeImage::LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels) {
    number_subimages = _number_subimages;
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    pixels = _pixels;
}


LargeImage* LargeImage::copy() const {
    return new LargeImage(number_subimages, channels, width, height, padding, classification, pixels);
}

int8_t LargeImage::get_pixel_unnormalized(int z, int y, int x) const {
    if (y < padding || x < padding) return 0;
    else if (y >= height + padding || x >= width + padding) return 0;
    else {
        return pixels[z][y - padding][x - padding];
    }
}

void LargeImage::set_pixel(int z, int y, int x, int8_t value) {
    if (y < padding || x < padding) return;
    else if (y >= height + padding || x >= width + padding) return;
    else {
        pixels[z][y - padding][x - padding] = value;
    }
}

#ifdef _HAS_TIFF_
void LargeImage::draw_image(string filename) const {
    // Open the TIFF file
    TIFF *output_image = NULL;

    cout << "drawing a TIFF with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    vector<uint8_t> values;
    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            //cout << "pushing back values for [" << y << "][" << x << "]!" << endl;
            values.push_back(get_pixel_unnormalized(0, y, x));
            values.push_back(get_pixel_unnormalized(1, y, x));
            values.push_back(get_pixel_unnormalized(2, y, x));
            //cout << "pushed back values for [" << y << "][" << x << "]!" << endl;
        }
    }

    if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL){
        std::cerr << "Unable to write tif file: " << "image.tiff" << std::endl;
    }

    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    // Write the information to the file

    tsize_t image_s;
    if ( (image_s = TIFFWriteEncodedStrip(output_image, 0, &values[0], 3 * (width + (padding * 2)) * (height + (padding * 2)) * sizeof(int8_t))) == -1) {
        std::cerr << "Unable to write tif file '" << filename << "'" << endl;
    }
    else {
        std::cout << "Image is saved to '" << filename << "', size is : " << image_s << endl;
    }

    TIFFWriteDirectory(output_image);
    TIFFClose(output_image);
}
#endif

int LargeImage::get_classification() const {
    return classification;
}

int LargeImage::get_number_subimages() const {
    return number_subimages;
}

int LargeImage::get_channels() const {
    return channels;
}

int LargeImage::get_height() const {
    return height;
}

int LargeImage::get_width() const {
    return width;
}

void LargeImage::get_pixel_avg(vector<float> &channel_avgs) const {
    channel_avgs.clear();
    channel_avgs.assign(channels, 0.0);

    //cout << "channels: " << channels << ", height: " << height << ", width: " << width << endl;

    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                channel_avgs[z] += pixels[z][y][x] / 255.0;
            }
        }
        channel_avgs[z] /= (height * width);
    }
}

void LargeImage::get_pixel_variance(const vector<float> &channel_avgs, vector<float> &channel_variances) const {
    channel_variances.clear();
    channel_variances.assign(channels, 0.0);

    float tmp;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                tmp = channel_avgs[z] - (pixels[z][y][x] / 255.0);
                channel_variances[z] += tmp * tmp;
            }
        }

        channel_variances[z] /= (height * width);
    }
}

void LargeImage::print(ostream &out) {
    out << "LargeImage Class: " << classification << endl;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                out << setw(7) << pixels[z][y][x];
            }
            out << endl;
        }
    }
}

string LargeImages::get_filename() const {
    return filename;
}

int LargeImages::read_images(string _filename) {
    filename = _filename;

    cout << "reading filename: " << filename << endl;

    ifstream infile(filename.c_str(), ios::in | ios::binary);

    if (!infile.is_open()) {
        cerr << "Could not open '" << filename << "' for reading." << endl;
        return 1;
    }

    int initial_vals[2];
    infile.read( (char*)&initial_vals, sizeof(initial_vals) );

    number_classes = initial_vals[0];
    number_images = initial_vals[1];

    cerr << "number_classes: " << number_classes << endl;
    cerr << "number_images: " << number_images << endl;


    class_sizes = vector<int>(number_classes, 0);

    for (int i = 0; i < number_images; i++) {
        int image_vals[4];
        infile.read( (char*)&image_vals, sizeof(image_vals) );

        int image_class = image_vals[0];
        channels = image_vals[1];
        int height = image_vals[2];
        int width = image_vals[3];

        cerr << "image[" << i << "] class: " << image_class << ", channels: " << channels << ", height: " << height << ", width: " << width << endl;

        class_sizes[image_class]++;

        int subimages_along_width = (width - subimage_width) + 1;
        int subimages_along_height = (height - subimage_height) + 1;
        int number_subimages = subimages_along_width * subimages_along_height;

        LargeImage large_image(infile, number_subimages, channels, width, height, padding, image_class, this);

        if (number_subimages < 0) {
            cerr << "ERROR! number subimages < 0!" << endl;
            continue;
        }

        images.push_back(large_image);
    }


    infile.close();

    cerr << "read " << images.size() << " images." << endl;
    for (int i = 0; i < (int32_t)class_sizes.size(); i++) {
        cerr << "    class " << setw(4) << i << ": " << class_sizes[i] << endl;
    }

    //update number images to number of subimages
    number_images = 0;
    for (int i = 0; i < images.size(); i++) {
        number_images += images[i].get_number_subimages();
    }

    cerr << "number_subimages: " << number_images << endl;

    /*
   for (int i = 0; i < images.size(); i++) {
       images[i].print(cerr);
   }
   */

    return 0;
}



LargeImages::LargeImages(string _filename, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev) {
    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filename = _filename;
    read_images(filename);

    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;

    for (int32_t j = 0; j < channels; j++) {
        cerr << "setting pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        cerr << "setting pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}

LargeImages::LargeImages(string _filename, int _padding, int _subimage_height, int _subimage_width) {
    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filename = _filename;
    read_images(filename);

    calculate_avg_std_dev();
}

int LargeImages::get_class_size(int i) const {
    return class_sizes[i];
}

int LargeImages::get_number_classes() const {
    return number_classes;
}

int LargeImages::get_number_images() const {
    return number_images;
}

int LargeImages::get_number_large_images() const {
    return images.size();
}

int LargeImages::get_number_subimages(int i) const {
    return images[i].get_number_subimages();
}



int LargeImages::get_image_channels() const {
    return channels;
}

int LargeImages::get_image_width() const {
    return subimage_width + (2 * padding);
}

int LargeImages::get_image_height() const {
    return subimage_height + (2 * padding);
}

int LargeImages::get_image_classification(int image) const {
    return images[image].get_classification();
}

int LargeImages::get_classification(int subimage) const {
    for (int32_t i = 0; i < images.size(); i++) {
        if (subimage < images[i].get_number_subimages()) {
            return images[i].get_classification();
        } else {
            subimage -= images[i].get_number_subimages();

            if (i == images.size() - 1) return images[i].get_classification();
        }
    }

    cerr << "Error getting classification, subimage was: " << subimage << " and there are not that many subimages!" << endl;
    exit(1);
    return 0;
}

float LargeImages::get_pixel(int subimage, int z, int y, int x) const {
    //cout << "getting pixel from subimage: " << subimage << ", z: " << z << ", y: " << y << ", x: " << x << endl;

    for (int32_t i = 0; i < images.size(); i++) {
        if (subimage < images[i].get_number_subimages()) {
            const LargeImage &image = images[i];

            if (y < padding || x < padding) return 0;
            else if (y >= subimage_height + padding || x >= subimage_width + padding) return 0;
            else {
                int subimages_along_width = image.get_width() - subimage_width + 1;

                int subimage_y_offset = subimage / subimages_along_width;
                int subimage_x_offset = subimage % subimages_along_width;

                return ((image.pixels[z][subimage_y_offset + y - padding][subimage_x_offset + x - padding] / 255.0) - channel_avg[z]) / channel_std_dev[z];
            }
        } else {
            subimage -= images[i].get_number_subimages();
        }
    }

    cerr << "Error getting normalized pixel, subimage was: " << subimage << " and there are not that many subimages!" << endl;
    exit(1);
    return 0;
}


const vector<float>& LargeImages::get_average() const {
    return channel_avg;
}

const vector<float>& LargeImages::get_std_dev() const {
    return channel_std_dev;
}

float LargeImages::get_channel_avg(int channel) const {
    return channel_avg[channel];
}

float LargeImages::get_channel_std_dev(int channel) const {
    return channel_std_dev[channel];
}

void LargeImages::calculate_avg_std_dev() {
    //cerr << "calculating averages and standard deviations for images" << endl;
    channel_avg.clear();
    channel_avg.assign(channels, 0.0);

    //cerr << "number images: " << number_images << endl;

    vector<float> image_avg;
    for (int32_t i = 0; i < images.size(); i++) {
        //cout << "getting pixel avg for image: " << i << endl;
        images[i].get_pixel_avg(image_avg);

        for (int32_t j = 0; j < channels; j++) {
            channel_avg[j] += image_avg[j];
            //cout << "images[" << i << "], channel_avg[" << j << "]: " << channel_avg[j] << endl;
        }

        //cerr << "image width: " << images[i].get_width() << endl;
    }

    //TODO: calculate average and standard deviation for varying sized images
    //may want to average this over total pixels instead of per image

    for (int32_t j = 0; j < channels; j++) {
        channel_avg[j] /= images.size();
        cerr << "average pixel value for channel " << j << ": " << channel_avg[j] << endl;
    }

    channel_std_dev.clear();
    channel_std_dev.assign(channels, 0.0);

    vector<float> image_variance;
    for (int i = 0; i < images.size(); i++) {
        //cout << "getting pixel variance for image: " << i << endl;
        images[i].get_pixel_variance(channel_avg, image_variance);

        for (int32_t j = 0; j < channels; j++) {
            channel_std_dev[j] += image_variance[j];
            //cout << "images[" << i << "], channel_std_dev[" << j << "]: " << channel_std_dev[j] << endl;
        }
    }

    for (int32_t j = 0; j < channels; j++) {
        channel_std_dev[j] /= images.size();
        cerr << "pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        channel_std_dev[j] = sqrt(channel_std_dev[j]);
        cerr << "pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}

LargeImage* LargeImages::copy_image(int i) const {
    return images[i].copy();
}


#ifdef _HAS_TIFF_

void LargeImages::draw_image(int i, string filename) const {
    images[i].draw_image(filename);
}

#endif


#ifdef LARGE_IMAGES_TEST
int main(int argc, char **argv) {
    int subimage_y = 32;
    int subimage_x = 32;
    int padding = 3;

    LargeImages large_images(argv[1], padding, subimage_y, subimage_x);

    cout << "number classes: " << large_images.get_number_classes() << endl;
    cout << "number images: " << large_images.get_number_images() << endl;

}
#endif

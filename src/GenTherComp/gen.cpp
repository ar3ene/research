#include <random>
std::mt19937_64 rng_engine; // default-constructed engine
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
std::normal_distribution<double> normal_dist;  // global object

#include <cmath>
#include <ctime>
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>

//minimal std imports
using std::ios;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;

char st[256];

//thermodynamic parameters
double kt=0.1;
double j2=1.0;
double j4=1.0;

//training parameters
int n_batch=1;
int n_cycles=100;
double alpha_0=1.0;

//time parameters (microseconds)
double tau;
double timestep=1e-3;
double trajectory_time=2.5;

double loss_scale=timestep/trajectory_time;

//mnist parameters
const int lx=28;
const int number_of_pixels=lx*lx;
const int number_of_symbols=3;

//registers
int symbol_type[number_of_symbols];
double symbol[number_of_symbols][number_of_pixels];

//computer parameters
const int number_of_hidden_spins=512;
const int number_of_spins=number_of_pixels+number_of_hidden_spins;
const int number_of_bonds = number_of_pixels*number_of_hidden_spins+(number_of_hidden_spins*(number_of_hidden_spins-1))/2;
const int number_of_parameters=number_of_spins+number_of_bonds;

//bond registers
int number_of_neighbors[number_of_spins];
int bond_constituents[number_of_bonds][2];
int neighbors[number_of_spins][number_of_spins];
int bond_number[number_of_spins][number_of_spins];

double spin[number_of_spins];
double delta_spin[number_of_spins];
double grad_potential[number_of_spins];
double parameters[number_of_parameters];
double parameter_gradients[number_of_parameters];

//for training
int q_train=0;
int q_forward=0;
int q_image_bias=0;
int digits_seen;
int chosen_training_digit;

double loss=0.0;

//for display
int q_display_trajectory=0;

// Simple PNG writer for 28x28 RGB images (no compression, minimal chunks)

int global_trajectory_index = 0;
int global_snapshot_count = 0;
const int number_of_snapshots = 10;

//functions void
void initialize(void);
void read_mnist(void);
void langevin_step(void);
void run_trajectory(void);
void train_computer(void);
void set_interactions(void);
void connection_field_grid(void);
void accumulate_gradients(int i);
void set_initial_noise_data(void);
void assess_denoised_digits(void);
void set_initial_image_data(int s1);
void make_mnist_picture(int digit_id);
void example_noising_trajectories(void);
void example_denoising_trajectories(void);
void write_png(const char* filename, uint8_t* rgb, int width, int height);
void draw_thick_line(uint8_t* img, int width, int height, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, double thickness);

//functions int
int bias(int i);

//functions double
double image_bias(int s0);
double gauss_rv(double sigma);


int main(void){

//seed the RN generator, read MNIST
initialize();

//noising trajectories
example_noising_trajectories();

//train computer
train_computer();

//plot some visit-hidden connection weights
connection_field_grid();

//some denoising trajectories
example_denoising_trajectories();

//denoised digits (at t=tf)
assess_denoised_digits();


return 0;

}

double gauss_rv(double sigma){

normal_dist = std::normal_distribution<double>(0.0, sigma);
return normal_dist(rng_engine);
    
}


void initialize(void){

//clean up
snprintf(st,sizeof(st),"rm report_*.*");
cout << st << endl;
system(st);

//seed RN generator
rng_engine.seed(static_cast<uint64_t>(time(NULL)));

//read MNIST data
read_mnist();

//set interactions
set_interactions();

}


void read_mnist(void){

int i,j;
double dummy;

cout << "reading mnist from mnist_data.dat" << endl;

ifstream infile("mnist_data.dat", ios::in);

if (!infile.is_open()) {
cerr << "Could not open mnist_data.dat" << endl;
exit(1);
}

for (j = 0; j < 3; j++) {

infile >> symbol_type[j];

for (i = 0; i < number_of_pixels; i++) {
infile >> dummy;
symbol[j][i] = dummy;
}

}

infile.close();

cout << "mnist read" << endl;

}

int bias(int i){

return i+number_of_bonds;

}

void set_initial_image_data(int s1){

//holders
int q_train_holder=q_train;
int q_forward_holder=q_forward;
int q_image_bias_holder=q_image_bias;
int q_display_trajectory_holder=q_display_trajectory;

//select training digit
chosen_training_digit = s1 % 3;

//start with zeros
for(int i=0;i<number_of_spins;i++){spin[i]=gauss_rv(0.0);}

//evolve to equilibrium with training bias
//forward trajectory of the untrained computer
tau=0.0;
q_train=0;
q_forward=1;
q_image_bias=1;
q_display_trajectory=0;
while(tau<trajectory_time){langevin_step();tau+=timestep;}

//reset flags
tau=0.0;

q_train=q_train_holder;
q_forward=q_forward_holder;
q_image_bias=q_image_bias_holder;
q_display_trajectory=q_display_trajectory_holder;



/*
if((q_train==1) && (digits_seen<5)){

cout << " digit number " << digits_seen << " is " << d1 << endl;
make_mnist_picture(d1);
digits_seen++;

}
*/

}


void set_initial_noise_data(void){

//holders
int q_train_holder=q_train;
int q_forward_holder=q_forward;
int q_image_bias_holder=q_image_bias;
int q_display_trajectory_holder=q_display_trajectory;

//start with noise
for(int i=0;i<number_of_spins;i++){spin[i]=gauss_rv(0.0);}

//forward trajectory of the untrained computer
tau=0.0;
q_train=0;
q_forward=1;
q_display_trajectory=0;
while(tau<trajectory_time){langevin_step();tau+=timestep;}

//get ready for denoising
tau=0.0;

q_train=q_train_holder;
q_forward=q_forward_holder;
q_image_bias=q_image_bias_holder;
q_display_trajectory=q_display_trajectory_holder;

}

//accumulate parameter gradients during noising
void accumulate_gradients(int i){

double q1=0.0;

if(i>=number_of_bonds){
// if i is a bias

int s=i-number_of_bonds;
q1=(-delta_spin[s]+grad_potential[s]*timestep)/(2.0*kt);
parameter_gradients[i]+=q1;

}
else{
//i is a bond

//get s1 and s2
int s1=bond_constituents[i][0];
int s2=bond_constituents[i][1];

q1=(-delta_spin[s1]+grad_potential[s1]*timestep)*spin[s2]/(2.0*kt);
q1+=(-delta_spin[s2]+grad_potential[s2]*timestep)*spin[s1]/(2.0*kt);
parameter_gradients[i]+=q1;

}

}


void langevin_step(void){

// q_forward==1: noising trajectory
// q_forward==0: denoising trajectory

for(int i=0;i<number_of_spins;i++){

double xi=spin[i];
double xi3=xi*xi*xi;
double force=0.0;

//biases
if(q_forward==0){force=-2*j2*xi-4*j4*xi3-parameters[bias(i)];}
if(q_forward==1){force=-2*j2*xi-4*j4*xi3+image_bias(i);}


//interactions
if(q_forward==0){
for(int n=0;n<number_of_neighbors[i];n++){
int j=neighbors[i][n];
if(i!=j){force-=parameters[bond_number[i][j]]*spin[j];}
}}

//noise
double noise=sqrt(2.0*timestep*kt)*gauss_rv(1.0);
//if((q_forward==1) && (q_train==1)){noise=sqrt(2.0*timestep*0.00)*gauss_rv(1.0);}
delta_spin[i]=timestep*force+noise;

}


//update spins
for(int i=0;i<number_of_spins;i++){spin[i]+=delta_spin[i];}

//calculate quantities needed for gradients (start from new coords)
if((q_forward==1) && (q_train==1)){

for(int i=0;i<number_of_spins;i++){

double xi=spin[i];
double xi3=xi*xi*xi;

grad_potential[i]=2*j2*xi+4*j4*xi3+parameters[bias(i)];

for(int n=0;n<number_of_neighbors[i];n++){
int j=neighbors[i][n];
if(i!=j){grad_potential[i]+=parameters[bond_number[i][j]]*spin[j];}
}

}}


//gradients
if((q_forward==1) && (q_train==1)){
for(int i=0;i<number_of_parameters;i++){accumulate_gradients(i);}
}

//loss
if((q_forward==1) && (q_train==1)){
for(int i=0;i<number_of_spins;i++){
loss+=loss_scale*(-delta_spin[i]+grad_potential[i]*timestep)*(-delta_spin[i]+grad_potential[i]*timestep)/(4.0*kt*timestep);
}
}

}


void make_lattice_picture(int index) {

    uint8_t rgb[lx * lx * 3];

    for (int i = 0; i < lx * lx; ++i) {

        double v = spin[i];

        // New scaling: piecewise linear for |v|<threshold, otherwise tanh
        double threshold = 1.0;
        double scale = 2.0;
        double t;
        if (fabs(v) < threshold)
            t = 0.5 + 0.5 * v * scale;
        else
            t = 0.5 + 0.5 * tanh(scale * v);
        // Clamp t to [0.0, 1.0]
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;

        // Map t=0 (white) to t=1 (blue)
        uint8_t r = (uint8_t)(255 * (1.0 - t));
        uint8_t g = (uint8_t)(255 * (1.0 - t));
        uint8_t b = 255;

        rgb[3 * i + 0] = r;
        rgb[3 * i + 1] = g;
        rgb[3 * i + 2] = b;
    }

    char fname[128];
    int adjusted_index = index + global_trajectory_index * number_of_snapshots;
    snprintf(fname, sizeof(fname), "report_snapshot_%02d.png", adjusted_index);

    write_png(fname, rgb, lx, lx);
}

// Minimal PNG writer (uncompressed, 24-bit RGB, 28x28)
#include <zlib.h>
void write_png(const char* filename, uint8_t* rgb, int width, int height) {
    // Use stb_image_write or a minimal PNG writer, but for now use a PPM as a placeholder
    // (replace with PNG writer if needed)
    // Write PPM (for debugging)
    // FILE* f = fopen(filename, "wb");
    // fprintf(f, "P6\n%d %d\n255\n", width, height);
    // fwrite(rgb, 1, width * height * 3, f);
    // fclose(f);

    // Write PNG using libpng or stb_image_write, but here is a minimal zlib-compressed PNG
    // Use system call with ImageMagick's convert if PNG library is not available
    // For now, use stb_image_write if available, otherwise fallback to PPM and convert
    // For now, use a temporary PPM and convert to PNG
    char ppmname[128];
    snprintf(ppmname, sizeof(ppmname), "%s.ppm", filename);
    FILE* f = fopen(ppmname, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb, 1, width * height * 3, f);
    fclose(f);
    char cmd[512];
    // Resize to 560x560 (28*20) using ImageMagick, then remove ppm
    snprintf(cmd, sizeof(cmd), "magick %s -resize 560x560 %s; rm %s", ppmname, filename, ppmname);
    system(cmd);
}

void run_trajectory(void){

tau=0.0;
int save_interval=int(trajectory_time/(number_of_snapshots*timestep));
int snapshot_count=0;

while(tau<trajectory_time){

langevin_step();

if((q_display_trajectory==1) && ((int(tau/timestep)%save_interval)==0) && (snapshot_count<number_of_snapshots)){
    make_lattice_picture(snapshot_count);
    snapshot_count++;
}

tau+=timestep;
}

if(q_display_trajectory==1){
    char cmd[512];
snprintf(cmd,sizeof(cmd),"magick montage report_snapshot_*.png -tile 10x1 -geometry +2+2 -background none -label '' -font Helvetica report_trajectory.png");    system(cmd);
}
}

void example_noising_trajectories(void){

int n_traj=3;

//flags
q_train=0;
q_forward=1;
q_image_bias=2; //diminishing image bias
q_display_trajectory=1;

for(int i=0;i<n_traj;i++){

    cout << i << endl;

    set_initial_image_data(i);
    global_trajectory_index = i;
    run_trajectory();

    char cmd[512];
    snprintf(cmd,sizeof(cmd),"mv report_trajectory.png report_trajectory_%d.png",i);
    system(cmd);

    // cleanup snapshot files after each trajectory
    snprintf(cmd,sizeof(cmd),"rm report_snapshot_*.png");
    system(cmd);
}

snprintf(st,sizeof(st),"magick montage report_trajectory_*.png -tile 1x%d -geometry +2+2 -background none -label '' -font Helvetica report_all_noising_trajectories.png",n_traj);
system(st);

// cleanup trajectory montage files
snprintf(st, sizeof(st), "rm report_trajectory*.png");
system(st);

q_display_trajectory=0;

}

void example_denoising_trajectories(void){

//load parameters from parameters.dat
snprintf(st,sizeof(st),"parameters.dat");
ifstream in1(st,ios::in);
for(int k=0;k<number_of_parameters;k++){in1 >> parameters[k];}

int n_traj=10;

//flags
q_train=0;
q_forward=0;
q_image_bias=0;
q_display_trajectory=1;

for(int i=0;i<n_traj;i++){

    cout << i << endl;

    set_initial_noise_data();
    global_trajectory_index = i;
    run_trajectory(); // this will generate a full trajectory with multiple snapshots
    // do not attempt to rename a nonexistent trajectory PNG
}

// create montage of denoising trajectories
snprintf(st,sizeof(st),"magick montage report_snapshot_*.png -tile 10x10 -geometry +2+2 -background none -label '' -font Helvetica report_all_denoising_trajectories.png");
system(st);

// cleanup snapshot files
snprintf(st,sizeof(st),"rm report_snapshot_*.png");
system(st);

// cleanup trajectory montage files
snprintf(st, sizeof(st), "rm report_trajectory*.png");
system(st);

q_display_trajectory=0;

}

void train_computer(void){

snprintf(st,sizeof(st),"parameters.dat");
ofstream out1(st,ios::out);

snprintf(st,sizeof(st),"report_loss.dat");
ofstream out2(st,ios::out);

snprintf(st,sizeof(st),"report_params.dat");
ofstream out3(st,ios::out);

//learning rate
double alpha=alpha_0*loss_scale/sqrt(1.0*n_batch);

//flags
q_train=1;
q_forward=1;
q_image_bias=2; //diminishing image bias
q_display_trajectory=0;

//counter
digits_seen=0;

for(int i=0;i<n_cycles;i++){

//zero gradient accumulator
loss=0.0;
for(int k=0;k<number_of_parameters;k++){parameter_gradients[k]=0.0;}

//accumulate parameter gradients
for(int j=0;j<n_batch;j++){set_initial_image_data(i*n_batch+j);run_trajectory();}

//update parameters
double lambda = 0.0; // regularization strength
for(int k=0;k<number_of_parameters;k++){

parameter_gradients[k]+=lambda*parameters[k];

// update all parameters except visible-unit biases

//visible-hidden connections
if(k<number_of_pixels*number_of_hidden_spins){
parameters[k]-=alpha*parameter_gradients[k];
}

//hidden-hidden connections
if((k>=number_of_pixels*number_of_hidden_spins) && (k<number_of_bonds)){
parameters[k]-=alpha*parameter_gradients[k];
}

//visible-unit biases (zeroed)
//if((k>number_of_bonds)&&(k<number_of_bonds+number_of_pixels)){
//parameters[k]-=alpha*parameter_gradients[k];
//}

//hidden-unit biases
if(k>=number_of_bonds+number_of_pixels){
parameters[k]-=alpha*parameter_gradients[k];
}


}

// print selected parameters for inspection
cout << i+1 << " sample parameters: ";
cout << parameters[0] << " " << parameters[number_of_bonds+number_of_pixels] << " ";
cout << " loss " << loss/(1.0*n_batch) << endl;
out2 << i+1 << " " <<  loss/(1.0*n_batch) << endl;

cout << endl;

}

//output parameters
for(int k=0;k<number_of_parameters;k++){out1 << parameters[k] << endl;}
for(int k=0;k<number_of_parameters;k++){out3 << k << " " << parameters[k] << endl;}


}

void set_interactions(void){

int bond_count=0;
int s1,s2;

//visible-hidden connections
for(int i=0;i<number_of_pixels;i++){
for(int j=0;j<number_of_hidden_spins;j++){

s1=i;
s2=number_of_pixels+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

cout << "V-H connections " << bond_count << " " << number_of_pixels*number_of_hidden_spins << endl;

//hidden-hidden connections
for(int i=0;i<number_of_hidden_spins;i++){
for(int j=0;j<i;j++){

s1=number_of_pixels+i;
s2=number_of_pixels+j;

//neighbor lists
neighbors[s1][number_of_neighbors[s1]]=s2;
neighbors[s2][number_of_neighbors[s2]]=s1;
number_of_neighbors[s1]++;
number_of_neighbors[s2]++;

//bonds
bond_number[s1][s2]=bond_count;
bond_number[s2][s1]=bond_count;
bond_constituents[bond_count][0]=s1;
bond_constituents[bond_count][1]=s2;
bond_count++;

}}

cout << "total connections " << bond_count << " " << number_of_bonds << endl;

//check J_{ij}=J_{ji}
for (int i = 0; i < number_of_spins; ++i){
for (int j = 0; j < number_of_spins; ++j){
if (bond_number[i][j] != bond_number[j][i]) {
std::cerr << "Asymmetry in bond_number at (" << i << ", " << j << ")" << std::endl;
}}}


}


void assess_denoised_digits(void){

    //load parameters from parameters.dat
    snprintf(st,sizeof(st),"parameters.dat");
    ifstream in1(st,ios::in);
    for(int k=0;k<number_of_parameters;k++){in1 >> parameters[k];}

    int n_traj=100;

    //flags
    q_train=0;
    q_forward=0;
    q_image_bias=0;
    q_display_trajectory=0;

    for(int i=0;i<n_traj;i++){

        cout << i << endl;

        set_initial_noise_data();
        run_trajectory();

        // also write PNG snapshot
        make_lattice_picture(i);
    }

    // generate montage image
    snprintf(st,sizeof(st),
        "magick montage report_snapshot_*.png -tile 10x10 -geometry +2+2 "
        "-background none -label '' -font Helvetica report_assessed_denoised_digits.png");
    system(st);

    // cleanup snapshot files
    snprintf(st,sizeof(st),"rm report_snapshot_*.png");
    system(st);
}

void draw_thick_line(uint8_t* img, int width, int height, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b, double thickness){

int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
int err = dx + dy;

while (true) {
    for (int ty = -int(thickness); ty <= int(thickness); ty++) {
    for (int tx = -int(thickness); tx <= int(thickness); tx++) {
        int px = x0 + tx;
        int py = y0 + ty;
        if (px >= 0 && px < width && py >= 0 && py < height) {
            int idx = 3 * (py * width + px);
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }}

    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
}
}

void make_mnist_picture(int digit_id) {

uint8_t rgb[lx * lx * 3];

for (int i = 0; i < number_of_pixels; ++i) {

    double v = symbol[digit_id][i];

    // scale to [0,1] using logistic for visual consistency
    double scale = 5.0;
    double t = 1.0 / (1.0 + exp(-scale * v));

    uint8_t r = (uint8_t)(255 * (1.0 - t));
    uint8_t g = (uint8_t)(255 * (1.0 - t));
    uint8_t b = 255;

    rgb[3 * i + 0] = r;
    rgb[3 * i + 1] = g;
    rgb[3 * i + 2] = b;
}

char fname[128];
snprintf(fname, sizeof(fname), "report_digit_%d.png", digit_id);

write_png(fname, rgb, lx, lx);
}


// For fabs, std::max
#include <cmath>
#include <algorithm>

void connection_field_grid(void) {

    //load parameters from parameters.dat
    snprintf(st,sizeof(st),"parameters.dat");
    ifstream in1(st,ios::in);
    for(int k=0;k<number_of_parameters;k++){in1 >> parameters[k];}

    // output image is 4x4 grid of 28x28 tiles
    const int tile_size = lx;
    const int grid_dim = 4;
    const int img_size = grid_dim * tile_size;

    uint8_t rgb[img_size * img_size * 3];

    // clear image to white
    for (int i = 0; i < img_size * img_size * 3; ++i) rgb[i] = 255;

    // randomly select 16 hidden units
    std::vector<int> chosen_units;
    while (chosen_units.size() < 16) {
        int j = number_of_pixels + std::uniform_int_distribution<int>(0, number_of_hidden_spins - 1)(rng_engine);
        if (std::find(chosen_units.begin(), chosen_units.end(), j) == chosen_units.end())
            chosen_units.push_back(j);
    }

    for (int idx = 0; idx < 16; ++idx) {
        int h = chosen_units[idx];

        // Compute once per block for efficiency
        int max_vis_neighbors = number_of_pixels;

        // determine min and max J_ij to normalize
        double jmin = 1e10, jmax = -1e10;
        for (int ni = 0; ni < max_vis_neighbors; ++ni) {
            int i = neighbors[h][ni];

            int b = bond_number[i][h];
            if (b >= 0) {
                double val = parameters[b];
                if (val < jmin) jmin = val;
                if (val > jmax) jmax = val;
            }
        }

        double jabs_max = std::max(fabs(jmin), fabs(jmax));

        // Visual separation for clarity
        for (int ni = 0; ni < max_vis_neighbors; ++ni) {
            int i = neighbors[h][ni];

            int b = bond_number[i][h];
            double val = 0.0;
            if (b >= 0) val = parameters[b];

            // diverging color: red for negative, blue for positive, darkness by magnitude
            double intensity = (jabs_max > 1e-12) ? fabs(val) / jabs_max : 0.0;
            if (intensity > 1.0) intensity = 1.0;
            uint8_t r, g, bl;
            if (val >= 0.0) {
                // blue channel ramps up
                r = (uint8_t)(255 * (1.0 - intensity));
                g = (uint8_t)(255 * (1.0 - intensity));
                bl = 255;
            } else {
                // red channel ramps up
                r = 255;
                g = (uint8_t)(255 * (1.0 - intensity));
                bl = (uint8_t)(255 * (1.0 - intensity));
            }

            int local_row = i / lx;
            int local_col = i % lx;

            int grid_row = idx / grid_dim;
            int grid_col = idx % grid_dim;

            int row = grid_row * tile_size + local_row;
            int col = grid_col * tile_size + local_col;

            int pix = 3 * (row * img_size + col);
            rgb[pix + 0] = r;
            rgb[pix + 1] = g;
            rgb[pix + 2] = bl;
        }
    }

    write_png("report_connection_fields.png", rgb, img_size, img_size);
}


double image_bias(int s0){

//no bias
if(q_image_bias==0){
return (0.0);
}

if(q_image_bias>0){

double p1;

if(s0<number_of_pixels){p1=2.0*(symbol[chosen_training_digit][s0]);}
else{p1=2.0*(symbol[chosen_training_digit][s0 % number_of_pixels]);}
//provide hidden spins with some image-specific structure during training

//fixed image, for initial equilibration
if(q_image_bias==1){
return (p1);
}

//diminishing image bias, for noising
if(q_image_bias==2){

double t1=tau/(0.75*trajectory_time);

if(t1>1.0){return 0.0;}
else{return (1.0-t1)*p1;}

}
}

return (0.0);

}

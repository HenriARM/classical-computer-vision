#define STEP_POS 0.04f
#define STEP_LIFETIME 0.25f
#define GRAVITY 1.25f
#define HORIZONTAL_SPEED 0.5f
#define VERTICAL_SPEED 1.0f
#define COLOR_CHANNELS 4
#define FADE_OUT 0.10f
#define ILLUMINATE 3.0f

// sizeof(Particle)==32
struct Particle {
	float2 speed;
	float2 pos;
	float2 randVal;
	float lifetime;
};

// init a particle
void init_particle(__global struct Particle* particle) {
	const float speedX = particle->randVal.s0 * HORIZONTAL_SPEED;
	const float speedY = VERTICAL_SPEED + particle->randVal.s1;
	particle->speed = (float2)(speedX, speedY);
	particle->pos = (float2)(0.5f, 0.0f);
	particle->lifetime = 0.0f;
}

// init particles
__kernel void init_particles(__global struct Particle* particles, __global float* randVals) {
	const int n = get_global_id(0);
	particles[n].randVal = (float2)(randVals[2 * n], randVals[2 * n + 1]);
	init_particle(particles + n);
}

// set or add pixel at given image position
void draw_pixel(__global int* img, int2 pos, int4 color, bool add) {
	// don't draw outside of image
	if(pos.x < 0 || pos.x >= WINDOW_SIZE || pos.y < 0 || pos.y >= WINDOW_SIZE) return;
	// either set or add color value (ignore 4th color channel)
	const int base = COLOR_CHANNELS * (pos.y * WINDOW_SIZE + pos.x);
	switch(add) {
	case true:
		atomic_add(img + base, color.x);
		atomic_add(img + base + 1, color.y);
		atomic_add(img + base + 2, color.z);
		break;
	case false:
		img[base] = color.x;
		img[base + 1] = color.y;
		img[base + 2] = color.z;
		break;
	}
}

// draw particles
__kernel void draw_particles(__global struct Particle* particles, __global int* img) {
	// compute position in image
	const int n = get_global_id(0);
	const int2 imgPos = (int2)(particles[n].pos.x * WINDOW_SIZE, WINDOW_SIZE - 1 - WINDOW_SIZE * particles[n].pos.y);
	// draw particle
	int radius = 3;
	for(int dx=-radius; dx<=radius; dx++) {
		for(int dy=-radius; dy<=radius; dy++) {
			const float dist = sqrt((float)(dx * dx + dy * dy));
			__global struct Particle* particle = particles + n;
			// BGR
			const float4 color = mix((float4)(1.0f, 1.0f, 0.0f, .0f), (float4)(1.0f, 1.0f, 1.0f, 0.0f), exp(-particle->lifetime)) * exp(-particle->lifetime * FADE_OUT) * exp(ILLUMINATE - dist);
			draw_pixel(img, imgPos + (int2)(dx, dy), convert_int4(color * 255.0f), true);
		}
	}
}

// update state of particles
__kernel void update_particles(__global struct Particle* particles) {
	const int n = get_global_id(0);
	// re-init particle if it is out of window (fallen)
	if(particles[n].pos.y < 0.0f) {
		init_particle(particles + n);
	}
	// update position, speed and lifetime
	particles[n].pos = particles[n].pos + particles[n].speed * STEP_POS;
	particles[n].speed.y = particles[n].speed.y - GRAVITY * STEP_POS;
	particles[n].lifetime += STEP_LIFETIME;
}

// clear canvas
__kernel void clear_canvas(__global int* img) {
	draw_pixel(img, (int2)(get_global_id(0), get_global_id(1)), (int4)(0, 0, 0, 0), false);
}

// limit color channel values to 255
__kernel void saturate(__global int4* img) {
	const int base = get_global_id(1) * WINDOW_SIZE + get_global_id(0);
	img[base] = min(img[base], (int4)(255, 255, 255, 0));
}
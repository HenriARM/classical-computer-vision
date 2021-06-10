import os
import random
import cv2
import numpy as np
import pyopencl as cl

WINDOW_SIZE = 480
COLOR_CHANNELS = 4  # RGBA
COLOR_CHANNEL_SIZE = 4  # we need int32 to perform atomic operations for multiple particles at same position)
PARTICLES_NUM = 400
PARTICLE_STRUCT_SIZE = 32  # sizeof(struct Particle)

KERNEL_PATH = './kernel.cl'


def main():
    # setup OpenCL
    platforms = cl.get_platforms()  # a platform corresponds to a driver (e.g. AMD)
    platform = platforms[0]  # take first platform
    devices = platform.get_devices(cl.device_type.GPU)  # get GPU devices of selected platform
    device = devices[0]  # take first GPU
    context = cl.Context([device])  # put selected GPU into context object
    queue = cl.CommandQueue(context, device)  # create command queue for selected GPU and context

    # setup buffer for particles
    particles_buff = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=PARTICLE_STRUCT_SIZE * PARTICLES_NUM,
                               hostbuf=None)

    # setup random values (for random speed and color)
    random.seed()
    rand_values = np.array([random.random() - 0.5 for _ in range(2 * PARTICLES_NUM)], dtype=np.float32)
    bufRandVals = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=rand_values)

    img = np.zeros([WINDOW_SIZE, WINDOW_SIZE, COLOR_CHANNELS], dtype=np.int32)  # must be square to ignore distortion
    img_buff = cl.Buffer(context, cl.mem_flags.WRITE_ONLY,
                         size=WINDOW_SIZE * WINDOW_SIZE * COLOR_CHANNELS * COLOR_CHANNEL_SIZE)

    # load and compile OpenCL program
    compilerSettings = f'-DWINDOW_SIZE={WINDOW_SIZE}'
    program = cl.Program(context, open(KERNEL_PATH).read()).build(compilerSettings)
    init_particles = cl.Kernel(program, 'init_particles')
    update_particles = cl.Kernel(program, 'update_particles')
    clear_canvas = cl.Kernel(program, 'clear_canvas')
    draw_particles = cl.Kernel(program, 'draw_particles')
    saturate = cl.Kernel(program, 'saturate')

    # init particles (https://documen.tician.de/pyopencl/runtime_program.html#pyopencl.enqueue_nd_range_kernel)
    init_particles.set_arg(0, particles_buff)
    init_particles.set_arg(1, bufRandVals)
    cl.enqueue_nd_range_kernel(queue, init_particles, (PARTICLES_NUM,), None)

    # since all particles start from same place, they will go all up
    for _ in range(100):
        update_particles.set_arg(0, particles_buff)
        cl.enqueue_nd_range_kernel(queue, update_particles, (PARTICLES_NUM,), None)
    while True:
        # clear canvas
        clear_canvas.set_arg(0, img_buff)
        cl.enqueue_nd_range_kernel(queue, clear_canvas, (WINDOW_SIZE, WINDOW_SIZE), None)

        # draw all particles
        draw_particles.set_arg(0, particles_buff)
        draw_particles.set_arg(1, img_buff)
        cl.enqueue_nd_range_kernel(queue, draw_particles, (PARTICLES_NUM,), None)

        # saturate
        saturate.set_arg(0, img_buff)
        cl.enqueue_nd_range_kernel(queue, saturate, (WINDOW_SIZE, WINDOW_SIZE), None)

        # update particles
        update_particles.set_arg(0, particles_buff)
        cl.enqueue_nd_range_kernel(queue, update_particles, (PARTICLES_NUM,), None)

        # copy result from GPU and show
        cl.enqueue_copy(queue, img, img_buff, is_blocking=True)
        cv2.imshow("press ESC to exit", img.astype(np.uint8))

        # exit with ESC
        keyPressed = cv2.waitKey(10)
        if keyPressed == 27:
            break


if __name__ == '__main__':
    main()

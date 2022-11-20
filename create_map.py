import numpy as np

def add_rectangle(arr, x, y, width, height, color):
    arr[x:x+width, y:y+height, :] = color
    return arr

def field_center(side, row, border, line_length, line_width, field_width, size_x):
    y = int(border+row*(field_width+line_width)+field_width/2)
    if side=='left':
        x = int(border+line_length/2)
    elif side=='right':
        x = int(size_x-border-line_length/2)
    return (x,y)

def draw_map(map_size, border, line_width, line_length, field_width, nb_lines, tymio_width, obstacles, end):
    black = (0,0,0)
    grey = (70,70,70)
    white = (255,255,255)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    yellow = (255,255,0)

    size_x = map_size[0]
    size_y = map_size[1]

    map_arr = np.zeros(map_size, np.uint8)
    map_arr[:,:,:] = white

    # draw a white border around the field
    map_arr = add_rectangle(map_arr, border, border, size_x-1-2*border, size_y-1-2*border, grey)

    # draw the lines of the parking spaces
    for i in range(nb_lines):
        map_arr = add_rectangle(map_arr, border, border+field_width+i*(field_width+line_width), line_length, line_width, yellow)
        map_arr = add_rectangle(map_arr, size_x-1-border-line_length, border+field_width+i*(field_width+line_width), line_length, line_width, yellow)

    # draw the end field
    endfield = field_center(end[0],end[1], border, line_length, line_width, field_width, size_x)
    map_arr = add_rectangle(map_arr, endfield[0]-tymio_width//2, endfield[1]-tymio_width//2, tymio_width, tymio_width, red)

    # draw the obstacles
    for obst in obstacles:
        obst_field = field_center(obst[0],obst[1], border, line_length, line_width, field_width, size_x)
        map_arr = add_rectangle(map_arr, obst_field[0]-tymio_width//2, obst_field[1]-tymio_width//2, tymio_width, tymio_width, blue)

    return map_arr

#map_arr = draw_map(map_size)
#map_arr = np.transpose(map_arr, (1,0,2))
#image = Image.fromarray(map_arr)
#image.show()
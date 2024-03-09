import tkinter as tk
from tkinter import filedialog as tkfd
import pdb
from new_main import run_sfm
from write_to_ply import write_to_ply_file
from try_creating_mesh import launch_meshing_window
main_win = None
def start_app():
    global main_win
    main_win = tk.Tk(className = "Chitragama")
    main_win.geometry("500x200")
    # img_label_names = tk.StringVar()
    # img_label_names.set("Selected Images : \n")
    
    #img_label = tk.Label(main_win, textvariable = img_label_names)
    img_label = tk.Frame(main_win)
    img_label_text = tk.Text(img_label)
    img_label_text.insert(tk.END, "Selected Images : \n")
    img_label_scroll = tk.Scrollbar(img_label, command=img_label_text.yview)
    img_label_text.config(yscrollcommand=img_label_scroll.set)

    img_label_text.pack(side=tk.LEFT, fill=tk.Y)
    img_label_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    #img_label.grid(row=3, column = 0)
    image_list = []
    def add_image():
        added_images = tkfd.askopenfilenames(title = "Add Images to Reconstruct",
                                             filetypes =
                                             [("JPEG images", "*.jpg;*.JPG;*.JPEG"),
                                              ("PNG images", "*.png;*.PNG")])
        for img in added_images:
            image_list.append(img)
            #img_label_names.set(img_label_names.get() + "\n" + img)
            img_label_text.insert(tk.END, "\n" + img)
    buttons_frame = tk.Frame(main_win)
    buttons_frame.pack()
    add_img_button = tk.Button(buttons_frame, text = "Add Images", command = add_image)
    add_img_button.pack(side='left')
    #add_img_button.grid(row=1, column = 0)

    
    run_button_var = tk.IntVar()
    run_button_var.set(0)
    def start_sfm():
        run_button_var.set(1)
        
    def goto_mesh_creation():
        mesh_file = tkfd.askopenfilename(title = "Point Cloud File to Create Mesh With", filetypes = [("PLY files", "*.ply")])
        main_win.destroy()
        launch_meshing_window(mesh_file)
        was_closed = True
        run_button_var.set(1)

    create_mesh_button = tk.Button(buttons_frame, text = "Goto Creating Mesh", command = goto_mesh_creation)
    
    #create_mesh_button.grid(row = 1, column = 2)
    
    run_sfm_button = tk.Button(buttons_frame, text = "Run SFM", command = start_sfm)
    run_sfm_button.pack(side='left')
    create_mesh_button.pack(side='left')
    #run_sfm_button.grid(row=1, column = 1)
    #main_win.mainloop()
    was_closed = False
    img_label.pack()
    main_win.wait_variable(run_button_var)
    if not was_closed:
        from tkinter import ttk
        progress_container = tk.Frame(main_win)
        progress_container.pack()
        #progress_container.grid(row=2, column = 1)
        progress_title = tk.Label(progress_container, text = "SFM Views Procressing Progress")
        progress_title.pack()
        progress_bar = ttk.Progressbar(progress_container, mode = "indeterminate")
        progress_bar.pack()
        main_win.update()
        progress_bar.start()
        object_points = run_sfm(image_list, progress_bar)
        progress_bar.stop()
        ply_file, camera_file = write_to_ply_file(object_points)
        if tk.messagebox.askyesno("Launch Mesh Generation", "Do you want to try creating mesh out of just created point cloud ?"):
            main_win.destroy()
            launch_meshing_window(ply_file)
    

if __name__== "__main__":
    start_app()

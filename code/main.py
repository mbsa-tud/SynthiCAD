import blenderproc as bproc
import os
import argparse
import numpy as np
import glob
import json
import re

# define arguments
parser = argparse.ArgumentParser()
parser.add_argument('object', help="Path to the .ply file")
parser.add_argument('output_dir', help="Path, where the output files will be saved")
args = parser.parse_args()

# initialize blenderproc
bproc.init()

# Open the JSON file and read its contents
with open('code/config.json', 'r') as file:
    data = file.read()

# Remove comment lines using a regular expression
data = re.sub(r'//.*?\n', '', data)

# Parse the remaining JSON data
config = json.loads(data)

# Access the values in the config file
num_images = config['num_images']
coco = config['coco']
hdf5 = config['hdf5']
dataset_split = config['dataset_split']
train_percent = config['train_percent']/100
val_percent = config['val_percent']/100
test_percent = config['test_percent']/100
img_width = config['img_width']
img_height = config['img_height']
light_color_rgb = config['light_color_rgb']
light_energy = config['light_energy']
bg_color_rgb = config['bg_color_rgb']
obj_color = config['obj_color_rgb']
focus_objects = config['focus_objects']
focus_obj = config['focus_obj']
border_threshold = config['border_threshold']

# split data into train/val/test sets
train_thresh = int(train_percent * num_images) - 1
val_thresh = int(train_percent * num_images + val_percent * num_images) - 1
test_thresh = int(train_percent * num_images + val_percent * num_images + test_percent * num_images) - 1

# create a new light
light = bproc.types.Light()
light.set_energy(light_energy)
light.set_color(light_color_rgb)

# create light plane from ceiling (optional - need adjustments w.r.t object position)
light_plane = bproc.object.create_primitive('PLANE', scale=[5, 5, 1], location=[0, 0, 13])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_plane_material.make_emissive(emission_strength=3.5, replace=True)
light_plane.replace_materials(light_plane_material)

# set the camera resolution
bproc.camera.set_resolution(img_width, img_height)

# set the background color to grey
bproc.renderer.set_world_background(bg_color_rgb, 1)

# extract the objects from the user path
list_of_object_paths = glob.glob(args.object)
loaded_obj = []  # create an empty list to store the loaded objects

# load the objects into the scene
for obj_path in list_of_object_paths:
    if focus_objects:
        # check if any of the focus_names is in the object path
        for name in focus_obj:
            if name in obj_path:
                # load the object into the scene
                objec = bproc.loader.load_obj(obj_path)
                loaded_obj.extend(objec)
    else:
        # load all objects into the  scene
        objec = bproc.loader.load_obj(obj_path)
        loaded_obj.extend(objec)

# define the object poses sampling function


def sample_pose(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.25, -0.25, 0], [-0.3, -0.3, 0])
    max = np.random.uniform([0.27, 0.28, 0.10], [0.32, 0.33, 0.10])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# initialise variables
objj = []
a = 0

for obj in loaded_obj:
    # set custom property category_id to all objects
    obj.set_cp("category_id", a + 1)
    a += 1
    objj.append(obj)

    # scaling the object size
    obj.set_scale([1.25, 1.25, 1.25])

    # Set the light location
    light_location = bproc.sampler.shell(obj.get_location(), radius_min=1, radius_max=1.5,
                                         elevation_min=5, elevation_max=89, uniform_volume=False)
    light.set_location(light_location)

    # Set the metallic properties for the object
    mat = obj.get_materials()[0]
    mat.set_principled_shader_value("Metallic", 1)

    # Creating bvh_tree for faster ray casting
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objj)

    # Initialize iterating variables
    i_poses = 0
    poses = 0
    tries = 0

    while tries < 1000000 and poses < num_images:
        # Reset keyframes
        bproc.utility.reset_keyframes()

        # Calling the object poses sampling function + collision check within the loaded objects in the scene
        bproc.object.sample_poses(
            objj,
            sample_pose_func=sample_pose,
            objects_to_check_collisions=None,
            max_tries=10000,
            mode_on_failure="last_pose",
        )

        # setting the camera location
        location = bproc.sampler.shell(center=[0, 0, 0],
                                       radius_min=0.61,
                                       radius_max=1.84,
                                       elevation_min=5,
                                       elevation_max=89,
                                       uniform_volume=False)

        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(objj) + np.random.uniform([-0.05, -0.05, 0], [0.05, 0.05, 0])

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=None)

        # Add homog cam pose based on location and rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Check that objects are at certain meters away from the camera and make sure the view interesting enough i.e objects are not placed too close to camera
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 1.0, "max": 4.0}, bvh_tree,
                                                       sqrt_number_of_rays=10):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix)
        else:
            tries += 1
            continue

        # Check if the object is visible within the camera's field of view
        visible_objects = bproc.camera.visible_objects(cam2world_matrix)
        for ob in objj:
            if ob not in visible_objects:
                tries += 1
                break
        else:
            # Initial bounding box checks for all objects within the image dimensions
            in_bounds = True
            for ob in objj:
                bbox = ob.get_bound_box()
                if bbox[0][0] < 0 or bbox[1][0] > img_width or bbox[0][1] < 0 or bbox[1][1] > img_height:
                    in_bounds = False
                    tries += 1
                    break

            if in_bounds:
                # Rendering Segmentation masks
                seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"],
                                                        default_values={"category_id": 0, "class_label": 'background'})

                # Extract class segmap and attribute maps from rendered segmentation masks
                attribute_maps = seg_data['instance_attribute_maps']

                # Extract category ids from attribute maps
                category_ids = []
                for attribute_map in attribute_maps:
                    for obj_attrs in attribute_map:
                        category_id = obj_attrs['category_id']
                        if category_id != 0:
                            category_ids.append(category_id)

                # Initialize dictionary and list to store bounding boxes and categorical masks of all objects
                bounding_boxes = {}
                cat_mask = []

                # Iterate over category ids and extract non-zero indices for each category
                for category_id in category_ids:
                    # Extract binary mask for specified category id
                    for segmap in seg_data['class_segmaps']:
                        cat_mask = np.zeros_like(segmap,
                                                 dtype=np.uint8)  # cat_mask to get the same shape as segmap with values zero as its elements
                        for instance_attr in attribute_maps:
                            for obj_attrs in instance_attr:
                                if obj_attrs['category_id'] == category_id:
                                    cat_mask += (segmap == obj_attrs['category_id']).astype(
                                        np.uint8)  # replacing the elements in cat_mask with category_id of each object

                    # Check if cat_mask has any non-zero pixels
                    if np.count_nonzero(cat_mask) == 0:
                        # Category not present in instance map
                        bounding_boxes[category_id] = None
                        tries += 1
                        continue
                    else:
                        # Get non-zero pixels of the cat_mask
                        non_zero_pixels = np.argwhere(cat_mask)
                        # Compute bounding box
                        bbox = np.array([[np.min(non_zero_pixels[:, 1]), np.min(non_zero_pixels[:, 0])],
                                         [np.max(non_zero_pixels[:, 1]), np.max(non_zero_pixels[:, 0])]])
                        bounding_boxes[category_id] = bbox

                # Check if all bounding boxes are contained within the output image without truncation

                all_truncated = True

                for category_id, bbox in bounding_boxes.items():
                    # Check if category is present in class segmap
                    if bbox is None:
                        continue

                    # Compute the width and height of the bounding box
                    bbox_width = bbox[1][0] - bbox[0][0] + 1
                    bbox_height = bbox[1][1] - bbox[0][1] + 1
                    # Compute the border x and y values
                    border_x = border_threshold * img_width
                    border_y = border_threshold * img_height

                    # Check if the bounding box is truncated
                    if bbox[0][0] < border_x or bbox[0][1] < border_y or \
                            bbox[1][0] >= img_width - border_x or bbox[1][1] >= img_height - border_y or \
                            bbox_width > img_width or bbox_height > img_height:
                        all_truncated = True
                        tries += 1
                        continue
                    else:
                        all_truncated = False

                # increment the camera poses if the objects are not truncated
                if not all_truncated:
                    poses += 1
                    # Render RGB images if desired number of objects are met
                    if objj.__len__() == len(loaded_obj):
                        data = bproc.renderer.render()
                        if coco:
                            if dataset_split:
                                # Define output directories and file prefixes
                                if i_poses <= train_thresh:
                                    output_dir = os.path.join(args.output_dir, 'coco_data', 'Train_Dataset')
                                elif i_poses <= val_thresh:
                                    output_dir = os.path.join(args.output_dir, 'coco_data', 'Valid_Dataset')
                                else:
                                    output_dir = os.path.join(args.output_dir, 'coco_data', 'Test_Dataset')

                                i_poses += 1

                                # Write data to COCO file
                                bproc.writer.write_coco_annotations(
                                    output_dir,
                                    instance_segmaps=seg_data["instance_segmaps"],
                                    instance_attribute_maps=seg_data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG",
                                    jpg_quality=100,
                                    append_to_existing_output=True,
                                    file_prefix='image_',
                                    indent=0,
                                )
                            else:
                                # Write data to COCO file
                                bproc.writer.write_coco_annotations(
                                    output_dir=os.path.join(args.output_dir, 'coco_data'),
                                    instance_segmaps=seg_data["instance_segmaps"],
                                    instance_attribute_maps=seg_data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG",
                                    jpg_quality=100,
                                    append_to_existing_output=True,
                                    file_prefix='image_',
                                    indent=0,
                                )

                        if hdf5:
                            # write the data to a .hdf5 container
                            bproc.writer.write_hdf5(args.output_dir + "/hdf5", data, append_to_existing_output=True)

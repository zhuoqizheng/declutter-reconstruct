from __future__ import division

import os
import time

import cv2

import util.visualize as visualize
import util.layout as layout
import util.evaluation as eval
import util.postprocessing as post
import util.voronoi as vr
from object import Segment as sg, ExtendedSegment
from object import Surface as Surface

import FFT_MQ as fft

EVAL_FLAG = False


def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def print_parameters(param_obj, path_obj):
    file_parameter = open(path_obj.filepath + 'parameters.txt', 'w')
    param = param_obj.__dict__
    for par in param:
        file_parameter.write(par + ": %s\n" % (getattr(param_obj, par)))


def final_routine(img_ini, param_obj, size, draw, extended_sgms_th1_merged,
                  ind, rooms_th1, filepath):
    post.clear_rooms(filepath + '8b_rooms_th1.png', param_obj, rooms_th1)

    if draw.rooms_on_map:
        segmentation_map_path = visualize.draw_rooms_on_map(
            img_ini, '8b_rooms_th1_on_map', size, filepath=filepath)

    if draw.rooms_on_map_prediction:
        visualize.draw_rooms_on_map_prediction(
            img_ini, '8b_rooms_th1_on_map_prediction', size, filepath=filepath)

    if draw.rooms_on_map_lines:
        visualize.draw_rooms_on_map_plus_lines(img_ini,
                                               extended_sgms_th1_merged,
                                               '8b_rooms_th1_on_map_th1' +
                                               str(ind),
                                               size,
                                               filepath=filepath)

    # -------------------------------------POST-PROCESSING------------------------------------------------
    segmentation_map_path_post, colors = post.oversegmentation(
        segmentation_map_path, param_obj.th_post, filepath=filepath)
    return segmentation_map_path_post, colors


def start_main(par, param_obj, path_obj):
    param_obj.tab_comparison = [[''], ['precision_micro'], ['precision_macro'],
                                ['recall_micro'], ['recall_macro'],
                                ['iou_micro_mean_seg_to_gt'],
                                ['iou_macro_seg_to_gt'],
                                ['iou_micro_mean_gt_to_seg'],
                                ['iou_macro_gt_to_seg']]
    start_time_main = time.time()

    draw = par.ParameterDraw()
    filepath = path_obj.filepath

    print_parameters(param_obj, path_obj)

    # ----------------------------1.0_LAYOUT OF ROOMS------------------------------------
    # ------ starting layout
    # read the original image
    orebro_img = cv2.imread(path_obj.orebro_img)
    width = orebro_img.shape[1]
    height = orebro_img.shape[0]
    size = [width, height]
    img_rgb = cv2.bitwise_not(orebro_img)
    # making a copy of original image
    img_ini = cv2.imread(path_obj.metric_map_path)

    # -----------------------------1.1_CANNY AND HOUGH-------------------------------------
    walls, _ = layout.start_canny_and_hough(img_rgb, param_obj)
    # vector of lines. line: \f$(x_1, y_1, x_2, y_2)\f$ , where \f$(x_1,y_1)\f$ and \f$(x_2, y_2)\f$ are the ending points of each detected line segment.

    if draw.map:
        visualize.draw_image(img_ini, '0_Map', size, filepath=filepath)
    if draw.hough:
        visualize.draw_hough(img_ini,
                             walls,
                             '2_Hough',
                             size,
                             filepath=filepath)

    lines = walls
    walls = layout.create_walls(lines)

    print("lines:", len(lines), 'walls:', len(walls))
    if not param_obj.bormann:
        if draw.walls:
            # draw Segments
            visualize.draw_walls(walls, '3_Walls', size, filepath=filepath)

        lim1, lim2 = 300, 450
        while not (lim1 <= len(walls) <= lim2):
            if param_obj.filter_level <= 0.12:
                break

            if len(walls) < lim1:
                param_obj.set_filter_level(param_obj.filter_level - 0.02)
            if len(walls) > lim2:
                param_obj.set_filter_level(param_obj.filter_level + 0.02)
            fft.main(path_obj.metric_map_path, path_obj.path_orebro,
                     param_obj.filter_level, param_obj)
            path_obj.orebro_img = filepath + 'OREBRO_' + str(
                param_obj.filter_level) + '.png'

    # ------1.2_SET XMIN, YMIN, XMAX, YMAX OF walls-----------------------------------
    # from all points of walls select x and y coordinates max and min.
    xmin, xmax, ymin, ymax = sg.find_wall_boudary(walls, param_obj.offset,
                                                  size)

    # -------1.3 EXTERNAL CONTOUR--------------------------------------------------
    img_cont = cv2.imread(path_obj.metric_map_path)
    (contours, vertices) = layout.external_contour(img_cont)
    if draw.contour:
        visualize.draw_contour(vertices, '4_Contour', size, filepath=filepath)

    # ------1.4_MEAN SHIFT TO FIND ANGULAR CLUSTERS-------------------------------

    _, walls, angular_clusters = layout.cluster_ang(
        walls, param_obj.h, param_obj.minOffset, diagonals=param_obj.diagonals)

    angular_clusters = layout.assign_orebro_direction(param_obj.comp, walls)

    if draw.angular_cluster:
        visualize.draw_angular_clusters(angular_clusters,
                                        walls,
                                        '5a_angular_clusters',
                                        size,
                                        filepath=filepath)


    # -----1.5_SPATIAL CLUSTERS--------------------------------------------------
    # TODO Valerio's method
    wall_clusters = layout.get_wall_clusters(walls, angular_clusters)

    wall_cluster_without_outliers = []
    for cluster in wall_clusters:
        if cluster != -1:
            wall_cluster_without_outliers.append(cluster)

    # now that I have a list of clusters related to walls, I want to merge those very close each other
    # obtain representatives of clusters (all except outliers)
    repre_sgms = layout.get_representatives(
        walls, wall_cluster_without_outliers)

    repre_sgms = sg.spatial_clustering(
        param_obj.sogliaLateraleClusterMura, repre_sgms)

    # now we have a set of Segments with correct spatial cluster, now set the others with same wall_cluster
    spatial_clusters = layout.new_spatial_cluster(walls,
                                                  repre_sgms,
                                                  param_obj)
    if draw.representative_sgms:
        visualize.draw_representative_sgms(repre_sgms,
                                               "5b_representative_sgms",
                                               size,
                                               filepath=filepath)
    if draw.spatial_wall_cluster:
        visualize.draw_spatial_wall_clusters(wall_clusters,
                                             walls,
                                             '5c_spatial_wall_cluster',
                                             size,
                                             filepath=filepath)
    if draw.spatial_cluster:
        visualize.draw_spatial_clusters(spatial_clusters,
                                        walls,
                                        '5d_spatial_clusters',
                                        size,
                                        filepath=filepath)

    # -------------------------------------------------------------------------------------

    # ------------------------1.6 EXTENDED_LINES-------------------------------------------

    (extended_lines,
     extended_sgms) = layout.extend_line(spatial_clusters, walls, xmin,
                                             xmax, ymin, ymax)

    if draw.extended_lines:
        make_folder(filepath, 'Extended_Lines')
        visualize.draw_extended_lines(extended_sgms,
                                      walls,
                                      '7a_extended_lines',
                                      size,
                                      filepath=filepath + '/Extended_Lines')

    extended_sgms = sg.set_weights(extended_sgms, walls)
    # this is used to merge together the extended_sgms that are very close each other.
    extended_sgms_merged = ExtendedSegment.merge_together(
        extended_sgms, param_obj.distance_extended_segment, walls)
    extended_sgms_merged = sg.set_weights(extended_sgms_merged, walls)
    # this is needed in order to maintain the extended lines of the offset STANDARD
    border_lines = layout.set_weight_offset(extended_sgms_merged, xmax,
                                            xmin, ymax, ymin)
    extended_sgms_th1_merged, ex_li_removed = sg.remove_less_representatives(
        extended_sgms_merged, param_obj.th1)

    if draw.extended_lines:
        visualize.draw_extended_lines(extended_sgms_merged,
                                      walls,
                                      '7a_extended_lines_merged',
                                      size,
                                      filepath=filepath + '/Extended_Lines')
        visualize.draw_extended_lines(extended_sgms_th1_merged,
                                      walls,
                                      '7a_extended_lines_th1_merged',
                                      size,
                                      filepath=filepath + '/Extended_Lines')
    
    lis = []
    for line in ex_li_removed:
        short_line = sg.create_short_ex_lines(line, walls, size,
                                              extended_sgms_th1_merged)
        if short_line is not None:
            lis.append(short_line)

    lis = sg.set_weights(lis, walls)
    lis, _ = sg.remove_less_representatives(lis, 0.1)
    for el in lis:
        extended_sgms_th1_merged.append(el)

    if draw.extended_lines:
        visualize.draw_extended_lines(extended_sgms_th1_merged,
                                      walls,
                                      '7a_extended_lines_merged_plus_small',
                                      size,
                                      filepath=filepath + '/Extended_Lines')

    # -------------------------------------------------------------------------------------

    # ---------------1.7_EDGES--------------------------------------------
    # creating edges as intersection between extended lines

    edges = sg.create_edges(extended_sgms)
    edges_th1 = sg.create_edges(extended_sgms_th1_merged)
    # sg.set_weight_offset_edges(border_lines, edges_th1)

    # -------------------------------------------------------------------------------------
    # ---------------------------1.8_SET EDGES WEIGHTS-------------------------------------
    edges = sg.set_weights(edges, walls)
    edges_th1 = sg.set_weights(edges_th1, walls)

    if draw.edges:
        make_folder(filepath, 'Edges')
        visualize.draw_edges(edges,
                             walls,
                             -1,
                             '7c_edges',
                             size,
                             filepath=filepath + '/Edges')
        visualize.draw_edges(edges,
                             walls,
                             param_obj.threshold_edges,
                             '7c_edges_weighted',
                             size,
                             filepath=filepath + '/Edges')
        visualize.draw_edges(edges_th1,
                             walls,
                             -1,
                             '7c_edges_th1',
                             size,
                             filepath=filepath + '/Edges')
        visualize.draw_edges(edges_th1,
                             walls,
                             param_obj.threshold_edges,
                             '7c_edges_th1_weighted',
                             size,
                             filepath=filepath + '/Edges')

    # -------------------------------------------------------------------------------------

    # ----------------------------1.9_CREATE CELLS-----------------------------------------

    cells_th1 = Surface.create_cells(edges_th1)

    # -------------------------------------------------------------------------------------

    # ----------------Classification of Faces CELLE-----------------------------------------------------

    # Identification of Cells/Faces that are Inside or Outside the map
    global centroid
    if par.metodo_classificazione_celle == 1:
        print("1.classification method: ", par.metodo_classificazione_celle)
        (cells_th1, cells_out_th1, cells_polygons_th1, indexes_th1,
         cells_partials_th1, contour_th1, centroid_th1,
         points_th1) = layout.classification_surface(
             vertices, cells_th1, param_obj.division_threshold)

    # -------------------------------------------------------------------------------------

    # ---------------------------POLYGON CELLS---------------------------------------------
    # TODO this method could be deleted. check. Not used anymore.
    (cells_polygons_th1, polygon_out_th1, polygon_partial_th1,
     centroid_th1) = layout.create_polygon(cells_th1, cells_out_th1,
                                           cells_partials_th1)

    if draw.cells_in_out:
        visualize.draw_cells(cells_polygons_th1,
                             polygon_out_th1,
                             polygon_partial_th1,
                             '8a_cells_in_out_partial_th1',
                             size,
                             filepath=filepath)

    # ----------------------MATRICES L, D, D^-1, ED M = D^-1 * L--------------------------

    (matrix_l_th1, matrix_d_th1, matrix_d_inv_th1,
     X_th1) = layout.create_matrices(cells_th1, sigma=param_obj.sigma)

    # -------------------------------------------------------------------------------------

    # ----------------DBSCAN TO FIND CELLS IN THE SAME ROOM-------------------------

    cluster_cells_th1 = layout.DB_scan(param_obj.eps, param_obj.minPts, X_th1,
                                       cells_polygons_th1)

    if draw.dbscan:
        colors_th1, fig, ax = visualize.draw_dbscan(cluster_cells_th1,
                                                    cells_th1,
                                                    cells_polygons_th1,
                                                    edges_th1,
                                                    contours,
                                                    '7b_DBSCAN_th1',
                                                    size,
                                                    filepath=filepath)

    # -------------------------------------------------------------------------------------

    # ----------------------------POLYGON ROOMS--------------------------------------------

    rooms_th1, spaces_th1 = layout.create_space(cluster_cells_th1, cells_th1,
                                                cells_polygons_th1)

    # -------------------------------------------------------------------------------------

    # searching partial cells
    border_coordinates = [xmin, ymin, xmax, ymax]
    # TODO check how many time is computed
    cells_partials_th1, polygon_partial_th1 = layout.get_partial_cells(
        cells_th1, cells_out_th1, border_coordinates)

    polygon_out_th1 = layout.get_out_polygons(cells_out_th1)

    if draw.sides:
        visualize.draw_sides(edges_th1,
                             '14a_sides_th1',
                             size,
                             filepath=filepath)

    if draw.rooms:
        visualize.draw_rooms(rooms_th1,
                             colors_th1,
                             '8b_rooms_th1',
                             size,
                             filepath=filepath)

    if draw.cells_in_out:
        visualize.draw_cells(cells_polygons_th1,
                             polygon_out_th1,
                             polygon_partial_th1,
                             '8a_cells_in_out_partial_th1_post',
                             size,
                             filepath=filepath)

    # ---------------------------------END LAYOUT------------------------------------------

    ind = 0

    segmentation_map_path_post, colors = final_routine(
        img_ini,
        param_obj,
        size,
        draw,
        extended_sgms_th1_merged,
        ind,
        rooms_th1,
        filepath=filepath)
    old_colors = []
    voronoi_graph, coordinates = vr.compute_voronoi_graph(
        path_obj.metric_map_path,
        param_obj,
        False,
        '',
        param_obj.bormann,
        filepath=filepath)
    while old_colors != colors and ind < param_obj.iterations:
        ind += 1
        old_colors = colors
        vr.voronoi_segmentation(voronoi_graph,
                                coordinates,
                                param_obj.comp,
                                path_obj.metric_map_path,
                                ind,
                                filepath=filepath)

        segmentation_map_path_post, colors = final_routine(
            img_ini,
            param_obj,
            size,
            draw,
            extended_sgms_th1_merged,
            ind,
            rooms_th1,
            filepath=filepath)
    # -------------------------------------EVALUATION------------------------------------------------
    if EVAL_FLAG:
        if os.path.exists(path_obj.gt_color):
            eval.evaluate_bormann(path_obj.metric_map_name,
                                  path_obj.name_folder_input, param_obj)
            eval.evaluation(path_obj.gt_color, segmentation_map_path_post,
                            param_obj, '4 geometric', True, filepath)

    # -------------------------------------------------------------------------------------

    end_time_main = time.time()

    # write the times on file
    time_file = filepath + '17_times.txt'

    with open(time_file, 'w+') as TIMEFILE:
        print("time for main is: ",
              end_time_main - start_time_main,
              file=TIMEFILE)

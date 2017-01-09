//#include "/Users/tcmoore3/group-code/common.hpp"
#include "/Users/yangah/Programs/group-code/common.hpp"
#include "/Users/yangah/Programs/group-code/misc-tim.hpp"
//#include "/Users/tcmoore3/group-code/misc-tim.hpp"
int main() {
    Trajectory traj("end.xtc", "xtc");
    std::cout << "Read trajectory with " << traj.n_frames() << " frames." << std::endl;
    system_info info("../system-info.txt"); 
    system("mkdir -p h-bonds"); 
    bool do_hists = true; 
    float ns_per_frame = 0.050;
    int start = 0;  // final 10 ns for profiles, 100 frames / 5 ns
    if(start < 0)
        start = traj.n_frames() + start;
    double n_anal = traj.n_frames() - start;
    std::map<int, double> n0_scat_key = misc_charmm::n0_scattering_lengths();
    float bin_width = 1.0;  // angstrom
    histogram_1D profile_hist(-traj.box.back().period[2]/2, 
            traj.box.back().period[2]/2, bin_width);
    H_Bonds t_hbonds; 
    t_hbonds.use_klein_criteria(3.5, 2.6, 30.0);
    for(int i = 0; i < 3; i++) {
        t_hbonds.load_molecule_groups("../../scripts/h-bond-inputs", info.v_resname[i]); 
    }
    std::map<std::string, H_Bonds> h_bonds; 
    h_bonds["top"] = t_hbonds; h_bonds["bottom"] = t_hbonds; h_bonds["all"] = t_hbonds; 
    histogram_1D angle_hist(0, 90, 1.0);
    std::vector<std::string> layers; layers.push_back("top"); layers.push_back("bottom");
    std::map<std::string, histogram_1D> profiles;
    std::map<std::string, std::map<std::string, histogram_1D> > angle_hists;
    std::map<std::string, std::map<std::string, histogram_1D> > layer_hists;
    profiles["chol"] = profile_hist; profiles["chol-headgroups"] = profile_hist; 
    profiles["ecer2-headgroups"] = profile_hist; profiles["ecer2-tails"] = profile_hist; 
    profiles["ucer2-tails"] = profile_hist; profiles["ucer2-headgroups"] = profile_hist;
    profiles["ucer2-unequal-tail-part"] = profile_hist;
    profiles["water"] = profile_hist; profiles["neutron-scattering"] = profile_hist;
    profiles["chol-rings"] = profile_hist; profiles["chol-tail"] = profile_hist;
    profiles["ffa-tail"] = profile_hist; profiles["ffa-headgroups"] = profile_hist;
    profiles["ffa"] = profile_hist; profiles["ucer2"] = profile_hist; 
    profiles["ecer2"] = profile_hist;
    profiles["charge"] = profile_hist; profiles["electrons"] = profile_hist; 
    layer_hists["top"]["all"] = profile_hist; layer_hists["bottom"]["all"] = profile_hist;
    coord_t area, normal, pos_x; 
    pos_x.push_back(1.0); pos_x.push_back(0.0); pos_x.push_back(0.0);
    normal.push_back(0.0); normal.push_back(0.0); normal.push_back(1.0);
    std::ofstream o_s2("end-data/s2.txt"); 
    o_s2 << "# frame s2_top s2_bottom" << std::endl;
    std::ofstream o_all_angles("end-data/all-angles.txt");
    o_all_angles << "# top err_top bottom err_bottom" << std::endl; 
    std::ofstream o_ucer2_rot("end-data/ucer2-rotation.txt"); 
    o_ucer2_rot << "# time theta1 theta2 .. thetaN" << std::endl; 
    std::ofstream o_ecer2_rot_top("end-data/top-ecer2-rotation.txt"); 
    o_ecer2_rot_top << "# time theta1 theta2 .. thetaN" << std::endl; 
    std::ofstream o_ecer2_rot_bottom("end-data/bottom-ecer2-rotation.txt"); 
    o_ecer2_rot_bottom << "# time theta1 theta2 .. thetaN" << std::endl; 
    std::ofstream o_chol_rot("end-data/chol-rotation.txt"); 
    o_chol_rot << "# time theta1 theta2 .. thetaN" << std::endl; 
    std::ofstream o_apt("end-data/area-per-tail.txt");
    std::ofstream o_apl("end-data/area-per-lipid.txt"); 
    std::ofstream o_actual_apl("end-data/area-per-lipid-actual.txt"); 
    o_apt << "# frame apt_top apt_bottom" << std::endl;
    o_apl << "# frame apl" << std::endl;
    o_apl << "# frame apl_actual" << std::endl;
    std::ofstream o_ecer2("end-data/ecer2-xy-displacement.txt"); 
    o_ecer2 << "# frame xy" << std::endl; 
    std::ofstream o_ucer2("end-data/ucer2-xy-displacement.txt"); 
    o_ecer2 << "# frame xy" << std::endl; 
    std::ofstream o_ffa("end-data/ffa-xy-displacement.txt"); 
    o_ecer2 << "# frame xy" << std::endl; 
    std::ofstream o_chol("end-data/chol-xy-displacement.txt"); 
    o_ecer2 << "# frame xy" << std::endl; 
    std::ofstream o_middle_chol("end-data/middle-chol.txt");
    o_middle_chol << "# time(ps) n_chol_middle" << std::endl;
    for(int frame = 0; frame < traj.n_frames(); frame++) {
        int n_middle_chol = 0;
        float time_ns = ns_per_frame * frame; 
        float box_area = traj.box[frame].period[0] * traj.box[frame].period[1]; 
        std::map<std::string, std::map<std::string, std::vector<gbb> > > tails; 
        std::vector<gbb> molecules = info.coordlist_to_gbbs(traj.xyz[frame]);
        for(int i = 0; i < molecules.size(); i++) {
            std::string fn = "/Users/tcmoore3/lipid-prototypes/charmm/"
                + molecules[i].smolecule_id + "/"; 
            misc_charmm::set_number(molecules[i]);
            misc_charmm::set_groups(molecules[i]); 
            molecules[i].load_all_topology_no_coord(fn);
            misc_charmm::set_electrons(molecules[i]); 
        }
        coord_t lipids_com = calc_total_com(molecules, "tip3p-pppm");
        std::map<std::string, std::vector<gbb> > lipids_by_layer; 
        for(int i = 0; i < molecules.size(); i++) {
            molecules[i].calc_com(); 
            if(molecules[i].v_com[2] > lipids_com[2]) {
                molecules[i].descriptors["layer"] = "top"; 
                lipids_by_layer["top"].push_back(molecules[i]); 
            }
            else {
                molecules[i].descriptors["layer"] = "bottom"; 
                lipids_by_layer["bottom"].push_back(molecules[i]); 
            }
        }
        std::map<std::string, coord_t> layer_coms; 
        layer_coms["top"] = calc_total_com(lipids_by_layer["top"], "tip3p-pppm");
        layer_coms["bottom"] = calc_total_com(lipids_by_layer["bottom"], "tip3p-pppm");
        for(auto it = h_bonds.begin(); it != h_bonds.end(); ++it) {
            it->second.calc_all(lipids_by_layer[it->first], traj.box[frame]); 
        }
        o_ecer2_rot_top << frame << " "; o_ecer2_rot_bottom << frame << " ";
        o_ucer2_rot << frame << " "; o_chol_rot << frame << " "; 
        o_ecer2 << frame << " "; o_ucer2 << frame << " "; 
        o_ffa << frame << " "; o_chol << frame << " "; 
        for(int i = 0; i < molecules.size(); i++) {
            if(do_hists && frame >= start) {
                for(int j = 0; j < molecules[i].v_type.size(); j++) {
                    double scattering_length = n0_scat_key.find(molecules[i].v_type[j])->second;
                    double z = molecules[i].v_coord[j][2] - lipids_com[2]; 
                    double n_electrons = molecules[i].v_electrons[j]; 
                    double charge = molecules[i].v_charge[j]; 
                    profiles["neutron-scattering"].insert(z, scattering_length);
                    profiles["electrons"].insert(z, n_electrons);
                    profiles["charge"].insert(z, molecules[i].v_charge[j]);
                }
            }
            if(molecules[i].smolecule_id == "ucer2-hairpin-charmm") {
                if(do_hists && frame >= start) {
                    add_groups_to_zhist(molecules[i], profiles["ucer2-headgroups"],
                            "headgroup", lipids_com[2]);
                    add_groups_to_zhist(molecules[i], profiles["ucer2-unequal-tail-part"],
                            "fa_tail_unequal", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], profiles["ucer2-tails"],
                            "all_tail_atoms", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], 
                            layer_hists[molecules[i].descriptors["layer"]]["all"],
                            "all", lipids_com[2]); 
                }
                coord_t lipid_vec = sub_coords(
                        molecules[i].v_coord[4], molecules[i].v_coord[85]); 
                lipid_vec[2] = 0; 
                o_ucer2_rot << calc_angle(lipid_vec, pos_x) << " "; 
                tails[molecules[i].descriptors["layer"]]["all"].push_back(
                        make_sub_gbb_from_group(molecules[i], "fa_tail_equal"));
                tails[molecules[i].descriptors["layer"]]["all"].push_back(
                        make_sub_gbb_from_group(molecules[i], "sph_tail"));
                tails[molecules[i].descriptors["layer"]]["ucer2-fa"].push_back(
                        make_sub_gbb_from_group(molecules[i], "fa_tail_equal")); 
                tails[molecules[i].descriptors["layer"]]["ucer2-sph"].push_back(
                        make_sub_gbb_from_group(molecules[i], "sph_tail")); 
                tails[molecules[i].descriptors["layer"]]["fa_tail_unequal"].push_back(
                        make_sub_gbb_from_group(molecules[i], "fa_tail_unequal")); 
                o_ucer2 << calc_xy_distance(molecules[i].v_com,
                        layer_coms[molecules[i].descriptors["layer"]]) << " "; 

            }
            else if(molecules[i].smolecule_id == "ecer2-hairpin"){
                if(do_hists && frame >= start) {
                    add_groups_to_zhist(molecules[i], profiles["ecer2-headgroups"],
                            "headgroup", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], profiles["ecer2-tails"],
                            "all_tail_atoms", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], 
                            layer_hists[molecules[i].descriptors["layer"]]["all"],
                            "all", lipids_com[2]); 
                }
                tails[molecules[i].descriptors["layer"]]["all"].push_back(
                        make_sub_gbb_from_group(molecules[i], "fa_tail"));
                tails[molecules[i].descriptors["layer"]]["all"].push_back(
                        make_sub_gbb_from_group(molecules[i], "sph_tail"));
                tails[molecules[i].descriptors["layer"]]["ecer2-sph"].push_back(
                        make_sub_gbb_from_group(molecules[i], "sph_tail"));
                tails[molecules[i].descriptors["layer"]]["ecer2-fa"].push_back(
                        make_sub_gbb_from_group(molecules[i], "fa_tail"));
                o_ecer2 << calc_xy_distance(molecules[i].v_com,
                        layer_coms[molecules[i].descriptors["layer"]]) << " "; 
                coord_t lipid_vec = sub_coords(
                        molecules[i].v_coord[4], molecules[i].v_coord[61]); 
                lipid_vec[2] = 0; 
                if (molecules[i].descriptors["layer"] == "top"){
                    o_ecer2_rot_top << calc_angle(lipid_vec, pos_x) << " "; 
                }
                else {
                    o_ecer2_rot_bottom << calc_angle(lipid_vec, pos_x) << " "; 
                }
            }
            else if(molecules[i].smolecule_id == "tip3p-pppm") {
                if(do_hists && frame >= start) {
                    add_groups_to_zhist(molecules[i], profiles["water"], "all",
                            lipids_com[2]); 
                }
            }
            else if(molecules[i].smolecule_id == "cholesterol") {
                if(do_hists && frame >= start) {
                    add_groups_to_zhist(molecules[i], profiles["chol-headgroups"],
                            "headgroup", lipids_com[2]);
                    add_groups_to_zhist(molecules[i], profiles["chol"],
                            "all", lipids_com[2]);
                    add_groups_to_zhist(molecules[i], profiles["chol-rings"],
                            "rings", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], profiles["chol-tail"],
                            "tail", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], 
                            layer_hists[molecules[i].descriptors["layer"]]["all"],
                            "all", lipids_com[2]); 
                }
                double chol_angle = calc_angle(gbb_to_director(molecules[i]), normal);
                if(chol_angle > 90.0) {
                    chol_angle = 180.0 - chol_angle;
                }
                if(abs(molecules[i].v_com[2] - lipids_com[2]) < 5 && chol_angle > 60.0) {
                    n_middle_chol++;
                }
                else { 
                    tails[molecules[i].descriptors["layer"]]["all"].push_back(
                            make_sub_gbb_from_group(molecules[i], "all"));
                    tails[molecules[i].descriptors["layer"]]["chol"].push_back(
                            make_sub_gbb_from_group(molecules[i], "all"));
                    coord_t lipid_vec = sub_coords(
                            molecules[i].v_coord[4], molecules[i].v_coord[16]); 
                    lipid_vec[2] = 0; 
                    o_chol_rot << calc_angle(lipid_vec, pos_x) << " "; 
                }
            }
            else if(molecules[i].smolecule_id == "ffac24") {
                if(do_hists && frame >= start) {
                    add_groups_to_zhist(molecules[i], profiles["ffa-headgroups"],
                            "headgroup", lipids_com[2]);
                    add_groups_to_zhist(molecules[i], profiles["ffa"],
                            "all", lipids_com[2]);
                    add_groups_to_zhist(molecules[i], profiles["ffa-tail"],
                            "rings", lipids_com[2]); 
                    add_groups_to_zhist(molecules[i], 
                            layer_hists[molecules[i].descriptors["layer"]]["all"],
                            "all", lipids_com[2]); 
                }
                tails[molecules[i].descriptors["layer"]]["all"].push_back(
                        make_sub_gbb_from_group(molecules[i], "all"));
                tails[molecules[i].descriptors["layer"]]["ffa"].push_back(
                        make_sub_gbb_from_group(molecules[i], "all"));
                o_ffa << calc_xy_distance(molecules[i].v_com,
                        layer_coms[molecules[i].descriptors["layer"]]) << " "; 
            }
        }
        o_s2 << time_ns << " " << nematic_order_gbbs(tails["top"]["all"]) << " " 
             << nematic_order_gbbs(tails["bottom"]["all"]) << std::endl; 
        o_all_angles << time_ns; 
        o_apt << time_ns;
        for(int i = 0; i < layers.size(); i++) {
            coord_t angle_list = calc_tilt_angles(tails[layers[i]]["all"], normal); 
            o_all_angles << " " << average(angle_list) << " " 
                << stdev(angle_list) / sqrt(angle_list.size());
            coord_t director = nematic_director_gbbs(tails[layers[i]]["all"]); 
            float director_angle = calc_angle(director, normal); 
            if(director_angle > 90.0) {
                director_angle = 180.0 - director_angle; 
            }
            o_apt << " " << box_area*cos(
                    director_angle*M_PI/180)/tails[layers[i]]["all"].size(); 
            if(frame >= start) {
                for(auto tail_it = tails[layers[i]].begin();
                         tail_it != tails[layers[i]].end(); ++tail_it) {
                    angle_list = calc_tilt_angles(tails[layers[i]][tail_it->first], normal);
                    if(!angle_hists[layers[i]].count(tail_it->first)){
                        angle_hists[layers[i]][tail_it->first] = angle_hist; 
                    }
                    angle_hists[layers[i]][tail_it->first].insert_list(angle_list);
                }
            }
        }
        o_ecer2 << std::endl; o_ucer2 << std::endl; o_ffa << std::endl; o_chol << std::endl; 
        o_all_angles << std::endl;
        o_apt << std::endl;
        float actual_apl = 2*box_area/(info.cumulative_molecules(3) - n_middle_chol); 
        o_apl << time_ns << " " << 2*box_area/info.cumulative_molecules(3) << std::endl;
        o_actual_apl << time_ns << " " << actual_apl << std::endl;
        o_middle_chol << time_ns << " " << n_middle_chol << std::endl; 
        o_ucer2_rot << std::endl; o_ecer2_rot_top << std::endl; o_chol_rot << std::endl;
        o_ecer2_rot_bottom << std::endl; 
        if(frame >= start) {
            area.push_back(box_area);
        }
        if(frame % 10 == 0) {
            std::cout << "Finished " << time_ns << " ns." << std::endl;
        }
    }
    double avg_bin_vol = average(area) * bin_width; 
    if(do_hists) {
        for(auto it = profiles.begin(); it != profiles.end(); ++it) {
            it->second.normalize(n_anal*avg_bin_vol);
            std::string filename = "end-data/profile_" + it->first + ".txt";
            it->second.print(filename); 
        }
    }
    for(auto layer_it = angle_hists.begin(); layer_it != angle_hists.end(); ++layer_it) {
        for(auto tail_it = layer_it->second.begin();
                tail_it != layer_it->second.end(); ++tail_it) {
            tail_it->second.normalize();
            std::string fn = "end-data/hist_" + layer_it->first + "-"
                + tail_it->first + "_angle.txt";
            tail_it->second.print(fn);
        }
    }
    for(auto layer_it = layer_hists.begin(); layer_it != layer_hists.end(); ++layer_it) {
        for(auto tail_it = layer_it->second.begin();
                tail_it != layer_it->second.end(); ++tail_it) {
            tail_it->second.normalize(n_anal*avg_bin_vol);
            std::string fn = "end-data/profile_" + layer_it->first + "-"
                + tail_it->first + ".txt";
            tail_it->second.print(fn);
        }
    }
    for(auto it = h_bonds.begin(); it != h_bonds.end(); ++it) {
        std::string fn = "h-bonds/" + it->first + "_" + "h-bond-data.txt";
        it->second.print_summary(fn); 
        it->second.print_series("h-bonds", it->first); 
    }
    return 0;
}

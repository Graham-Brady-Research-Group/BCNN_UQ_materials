% This scripts reads the .fil files and stores coordinates, dispalcements,
% stresses etc based on the Abaqus2Matlab package and from the results
% derived from the nodal data from the .dat files
specs = load('specifications_2D.mat')
data_path = specs.data_path
for i = 0:specs.num_samples-1
    i
    data_path = [specs.data_path,'/sample_',num2str(i)];
    Rec = Fil2str([data_path,'/trimesh_',num2str(i),'.fil'])
    stress = readtable([data_path,'/stress_',num2str(i)]);
    % stress = stress(:,[1 2 4 5 6 7]);
    el_info = Rec1900(Rec);
    coordinate_node = Rec107(Rec);
    coordinate_node = array2table(coordinate_node);
    coordinate_node.Properties.VariableNames = {'Node','X', 'Y'};
    stress_node = [table2array(coordinate_node) zeros(size(table2array(coordinate_node),1),4)];
    for j = 1:size(stress,1)
        nd = table2array(stress(j,2));
        stress_node(nd,4:7) = table2array(stress(j,4:7));
    end
    stress_node = array2table(stress_node, 'VariableNames', {'Node','X', 'Y', 'S11', 'S22', 'S12', 'Mises',});
    delete(fullfile([data_path,'/stress_',num2str(i),'.csv']))
    writetable(stress,[data_path,'/stress_',num2str(i),'.csv'])
    writetable(stress_node,[data_path,'/stress_node_',num2str(i),'.csv'])
    writetable(cell2table(el_info),[data_path,'/el_info_',num2str(i),'.csv']) % element info
    %% This mainly concern integration points, will not save them for now
    stresses_int = Rec11(Rec);
    mises_int = Rec12(Rec);
    if isempty(mises_int)
        mises_int = zeros(size(stresses_int,1),1);
    else
        mises_int = mises_int(:,1);
    end
    stresses_data = [stresses_int mises_int];
    coordinates_int = Rec8(Rec);
    displacements = Rec101(Rec);
    displacements = displacements(:,2:end);
    coord_disp = Rec107(Rec);
    coord_disp = coord_disp(:,2:end);
    stresses_int_header = {'S11', 'S22', 'S12', 'Mises'};
    coordinates_header = {'X', 'Y'};
    displacements_header = {'U_x','U_y'};
    coords_disp_header = {'X_U', 'Y_U'};
    stress_node_bound_header = {'X', 'Y', 'S11', 'S22', 'S12', 'Mises'};

    zero_indices = [find(~table2array(stress_node(:, 2)));find(~table2array(stress_node(:, 3)));...
        find(table2array(stress_node(:, 2)) == 10);find(table2array(stress_node(:, 3)) == 10)];
    stress_int_bound = table2array(stress_node(zero_indices,2:end));
    stress_int_bound = [stress_int_bound;[coordinates_int stresses_data]];
    % Extract the second to the seventh column of these rows
    result = stress_node(zero_indices, 2:7);

    coordinates_table = array2table(coordinates_int, 'VariableNames', coordinates_header); % integration point coordinates

    coords_disp_table = array2table(coord_disp, 'VariableNames', coords_disp_header); % node coordinates
    displacements_table = array2table(displacements, 'VariableNames', displacements_header);% node displacements
    stresses_table = array2table(stresses_data, 'VariableNames', stresses_int_header); % integration point stresses
    stres_int_bound_table = array2table(stress_int_bound,'VariableNames', stress_node_bound_header);
    % writetable(coordinates_table, [data_path,'/coordinates_int',num2str(i),'.csv']);
    writetable(coords_disp_table, [data_path,'/coords_disp_',num2str(i),'.csv']);
    writetable(displacements_table, [data_path,'/disp_',num2str(i),'.csv']);
    % writetable(stresses_table, [data_path,'/stress_int_',num2str(i),'.csv']);
    % writetable(coordinate_node_table,[data_path,'/coordinate_node_',num2str(i),'.csv'])
    writetable(stres_int_bound_table, [data_path,'/stress_int_bound_',num2str(i),'.csv']);
end
%% Add path and load data
path = 'C:\Users\tien-\OneDrive\Desktop\Github\activation-patterns\data\sargolini\';
addpath(path)

pos = load([path '10073-17010302_POS.mat']);
t1c1 = load([path '10073-17010302_T1C1.mat']);
t2c1 = load([path '10073-17010302_T2C1.mat']);

%% Plot trajectories and spikes
% Square box: 100 x 100 x 50 cm
% Resolution: 50 Hz -> 10 mins

% convert vector of time stamps (in s) into a vector of time indices
time_idx = zeros(length(t2c1.cellTS), 1);
for i = 1:length(t2c1.cellTS)
    time_idx(i) = find(pos.post >= t2c1.cellTS(i), 1);
end

figure; plot(pos.posx, pos.posy) 
hold on;
scatter(pos.posx(time_idx), pos.posy(time_idx), 'red', 'filled')
set(gca,'xtick',[],'ytick',[]);

%% Convert absolute x-y trajectories into velocities
mins = 10; % duration in min
T = mins * 50 * 60; % duration in s

dt = comp_delta(pos.post);
dx = comp_delta(pos.posx);
dy = comp_delta(pos.posy);

vel_x = dx ./ dt; % cm/s
vel_y = dy ./ dt; % cm/s

figure; plot(vel_x)
figure; plot(vel_y)

% save as data matrix
save_mat = {pos.posx(1:T), pos.posy(1:T), vel_x(1:T-1), vel_y(1:T-1)};
save(['data/t2c1_' int2str(mins * 60) '.mat'], 'save_mat')

%% Plot grid cell map for 1 min data
posx_T = pos.posx(1:T);
posy_T = pos.posy(1:T);
time_idx_T = time_idx(time_idx <= size(posx_T, 1));

figure; plot(posx_T, posy_T) 
hold on;
scatter(posx_T(time_idx_T), posy_T(time_idx_T), 'red', 'filled')
set(gca,'xtick',[],'ytick',[]);
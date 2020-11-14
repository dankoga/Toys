function  dynamic_lorenz_gpu()
% The RK4 algorithm, in its cleanest version, needs a lot of temporary
% matrixes to store the system's state during different steps making it a
% very memory expensive method. So, to make it possible to run it in my
% 2GB RAM GPU, I had to make some awkward choices while coding. The present
% version avoids some temporary matrixes and some slow copy operations
% but be advised its not the best nor the cleanest code possible!

%--VIDEO PARAMETERS--%
% I choose the 1080p resolution and 60 fps.
% But, or some reason or other, my video editor recognized just 30 fps.
% Don't know why and I won't dig it...
Dimensions = 1080*[1, 1];
FILENAME = 'filename';
FrameRate = 60;
%--VIDEO PARAMETERS--%


%--CONSTANTS--%
sigma = 10.0;
rho   = 28.0;
beta  = 8.0/3.0;
LimitsZ = [  0.0, 50.0];
LimitsX = [-25.0, 25.0];
%--CONSTANTS--%

%--SYSTEM PARAMETERS--%
% 3333333 is about 10000000/3. The particles will be distribuited around the
% center in a normal distributon with sigma = ParticleSpread. I choose 3, just
% because... It was pretty.
ParticleNumber = 3333333;
ParticleSpread = 3;

% These are, in order, the initial conditions for the centers I used in the video.
% Uncomment anyone or use any other you like.

% ParticleInitialPositionR = [ 10.0,   0.0,  10.0];
% ParticleInitialPositionG = [  0.0,   0.0,  40.0];
% ParticleInitialPositionB = [-10.0,   0.0,  10.0];

% ParticleInitialPositionR = [ 10.0,   0.0,  40.0];
% ParticleInitialPositionG = [  0.0,   0.0,  10.0];
% ParticleInitialPositionB = [-10.0,   0.0,  40.0];

%These conditions are centerede around the critical points.
% p = rho - 1;
% b = sqrt(beta*p);
% ParticleInitialPositionR = [    b,     b,     p];
% ParticleInitialPositionG = [  0.0,   0.0,   0.0];
P% articleInitialPositionB = [   -b,    -b,     p];
%--SYSTEM PARAMETERS--%

%--INTEGRATION PARAMETERS--%
dt = 0.002;
IterationsMax = 10000;
%--INTEGRATION PARAMETERS--%

%--PRECOMPUTED CONSTANTS--%
dt_2 = dt/2.0;
dt_3 = dt/3.0;
dt_6 = dt/6.0;
gpuParticlesR = gpuArray([true(ParticleNumber, 1); ...
                          false(2*ParticleNumber, 1)]);
gpuParticlesG = gpuArray([false(ParticleNumber, 1); ...
                          true(ParticleNumber, 1); ...
                          false(ParticleNumber, 1)]);
gpuParticlesB = gpuArray([false(2*ParticleNumber, 1); ...
                          true(ParticleNumber, 1)]);
factorZ = Dimensions(1)/(LimitsZ(2) - LimitsZ(1));
sumZ = sum(LimitsZ);
factorX = Dimensions(2)/(LimitsX(2) - LimitsX(1));
%--PRECOMPUTED CONSTANTS--%

%--FUNCTIONS--%
lorenz = @(X) [sigma*(X(:,2) - X(:,1)), ...
               X(:,1).*(rho - X(:,3)) - X(:,2), ...
               X(:,1).*X(:,2) - beta*X(:,3)];
%--FUNCTIONS--%

%--GRAPHICAL OBJECTS--%
% vid is my archival lossless version. Ignore it.
% vid = VideoWriter(FILENAME,'Uncompressed AVI');
vid2 = VideoWriter(FILENAME,'MPEG-4');

% vid.FrameRate = FrameRate;
vid2.FrameRate = FrameRate;
vid2.Quality = 100;

% open(vid);
open(vid2);
%--GRAPHICAL OBJECTS--%


% Creating the two particles' position gpu arrays.
% OBS: The two arrays will alternate as the current and old positions because memory
%  restrictions. More details below.
gpuPosition = gpuArray(zeros(3*ParticleNumber, 3, 2));
gpuPosition(:,:,1) = ([ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionR; ...
                       ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionG; ...
                       ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionB]);

% Converting to Screen coordinates and checking bounds.
gpuScreen = round([(sumZ - gpuPosition(:,3,1))*factorZ, ...
                   (gpuPosition(:,1,1) - LimitsX(1))*factorX]);
gpuScreenValid = (gpuScreen(:,1) > 0) & (gpuScreen(:,1) <= Dimensions(1)) & ...
                 (gpuScreen(:,2) > 0) & (gpuScreen(:,2) <= Dimensions(2));

% Building 2D histograms.
gpuHistogramR = accumarray(gpuScreen(gpuParticlesR & gpuScreenValid,:,1), 1, Dimensions);
gpuHistogramG = accumarray(gpuScreen(gpuParticlesG & gpuScreenValid,:,1), 1, Dimensions);
gpuHistogramB = accumarray(gpuScreen(gpuParticlesB & gpuScreenValid,:,1), 1, Dimensions);

% Creating and recording image.
% The following normalization function was built "experimentally" using data
% from previous runs. It generates some pretty imagest I used for the first
% two cases, but the G channel is usually problematic due to  the initial
% conditions. After some experiments I decided to use a window average of 60
% frames for the third video. To repoduce the first two videos, just use the
% following functions for the three channels.
% gpuNormalization = 0.005 + 0.020*tanh((3e-4));
gpuDensity = gpuArray([max(gpuHistogramR(:))*ones(60,1), ...
                       mean(gpuHistogramG(:))*ones(60,1), ...
                       max(gpuHistogramB(:))*ones(60,1)]);
                     
% tanh(x) is a really good normalization and constraining function for images
% as it is limited to 1.0 for positive values  and has sigmoidal shape.
gpuNormalization = 1./gpuDensity(1,:);
gpuImage = cat(3, tanh(gpuHistogramR*gpuNormalization(1)), ...
                  tanh(gpuHistogramG*gpuNormalization(2)), ...
                  tanh(gpuHistogramB*gpuNormalization(3)));
% writeVideo(vid, gather(gpuImage));
writeVideo(vid2, gather(gpuImage));

format long;
for iteration = 1:IterationsMax
    % Integrating using RK4.
    % OBS: It was programmed this way to avoid temporary K matrixes.
    %      The arrays A and B alternate as the current and old positions.
    a = rem(iteration - 1,2) + 1;
    b = rem(iteration,2) + 1;
    gpuK = lorenz(gpuPosition(:,:,a));
    gpuPosition(:,:,b) = gpuPosition(:,:,a) + dt_6*gpuK;
    gpuK = lorenz(gpuPosition(:,:,a) + dt_2*gpuK);
    gpuPosition(:,:,b) = gpuPosition(:,:,b) + dt_3*gpuK;
    gpuK = lorenz(gpuPosition(:,:,a) + dt_2*gpuK);
    gpuPosition(:,:,b) = gpuPosition(:,:,b) + dt_3*gpuK;
    gpuK = lorenz(gpuPosition(:,:,a) + dt*gpuK);
    gpuPosition(:,:,b) = gpuPosition(:,:,b) + dt_6*gpuK;
    
    % Converting to Screen coordinates and checking bounds.
    gpuScreen = round([(sumZ - gpuPosition(:,3,b))*factorZ, ...
                       (gpuPosition(:,1,b) - LimitsX(1))*factorX]);
    gpuScreenValid = (gpuScreen(:,1) > 0) & (gpuScreen(:,1) <= Dimensions(1)) & ...
                     (gpuScreen(:,2) > 0) & (gpuScreen(:,2) <= Dimensions(2));
	% Building 2D histograms.
    gpuHistogramR = accumarray(gpuScreen(gpuParticlesR & gpuScreenValid,:), 1, Dimensions);
    gpuHistogramG = accumarray(gpuScreen(gpuParticlesG & gpuScreenValid,:), 1, Dimensions);
    gpuHistogramB = accumarray(gpuScreen(gpuParticlesB & gpuScreenValid,:), 1, Dimensions);
    % Creating and recording image.
%     gpuNormalization = 0.005 + 0.020*tanh((3e-4)*iteration);
    gpuDensity =[[max(gpuHistogramR(:)), mean(gpuHistogramG(:)), max(gpuHistogramB(:))]; ...
                 gpuDensity(1:end-1,:)];
    gpuNormalization = 1./mean(gpuDensity(:));
    gpuImage = cat(3, tanh(gpuHistogramR*gpuNormalization), ...
                      tanh(gpuHistogramG*gpuNormalization), ...
                      tanh(gpuHistogramB*gpuNormalization));
%     writeVideo(vid, gather(gpuImage));
    writeVideo(vid2, gather(gpuImage));
    
    [iteration IterationsMax]
end

% close(vid);
close(vid2);
close all;
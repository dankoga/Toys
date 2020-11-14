function  dynamic_lorenz_gpu_d()

sigma = 10.0;
rho   = 28.0;
beta  = 8.0/3.0;
%--SYSTEM PARAMETERS--%
ParticleNumber = 3333333;
ParticleSpread = 3;
p = rho - 1;
b = sqrt(beta*p);
ParticleInitialPositionR = [    b,     b,     p];
ParticleInitialPositionG = [  0.0,   0.0,   0.0];
ParticleInitialPositionB = [   -b,    -b,     p];
%--SYSTEM PARAMETERS--%

%--INTEGRATION PARAMETERS--%
dt = 0.002;
IterationsMax = 10000;
%--INTEGRATION PARAMETERS--%

%--VIDEO PARAMETERS--%
Dimensions = 1080*[1, 1];
DensityCutOff = 1.0;
%--VIDEO PARAMETERS--%

%--CONSTANTS--%


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

LimitsZ = [  0.0, 50.0];
LimitsX = [-25.0, 25.0]*Dimensions(2)/Dimensions(1);
deltaZ = Dimensions(1)/(LimitsZ(2) - LimitsZ(1));
sumZ = sum(LimitsZ);
deltaX = Dimensions(2)/(LimitsX(2) - LimitsX(1));
%--CONSTANTS--%

%--FUNCTIONS--%
lorenz = @(X) [sigma*(X(:,2) - X(:,1)), ...
               X(:,1).*(rho - X(:,3)) - X(:,2), ...
               X(:,1).*X(:,2) - beta*X(:,3)];
%--FUNCTIONS--%

% Creating the two particles' position gpu arrays:
% OBS: The two arrays will alternate as the current and old positions.
gpuPosition = gpuArray(zeros(3*ParticleNumber, 3, 2));
gpuPosition(:,:,1) = ([ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionR; ...
                       ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionG; ...
                       ParticleSpread*randn(ParticleNumber, 3) + ParticleInitialPositionB]);

Densities = zeros(IterationsMax,3);
format long;
for iteration = 1:IterationsMax
    % Integrating using RK4:
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
    
    % Converting to Screen coordinates and checking bounds:
    gpuScreen = round([(sumZ - gpuPosition(:,3,b))*deltaZ, ...
                       (gpuPosition(:,1,b) - LimitsX(1))*deltaX]);                      
    gpuScreenValid = (gpuScreen(:,1) > 0) & (gpuScreen(:,1) <= Dimensions(1)) & ...
                     (gpuScreen(:,2) > 0) & (gpuScreen(:,2) <= Dimensions(2));
	% Building 2D histograms:
    gpuHistogramR = accumarray(gpuScreen(gpuParticlesR & gpuScreenValid,:), 1, Dimensions);
    gpuHistogramG = accumarray(gpuScreen(gpuParticlesG & gpuScreenValid,:), 1, Dimensions);
    gpuHistogramB = accumarray(gpuScreen(gpuParticlesB & gpuScreenValid,:), 1, Dimensions);
    
    gpuNormalization = DensityCutOff./[max(gpuHistogramR(:)); ...
                                       max(gpuHistogramG(:)); ...
                                       max(gpuHistogramB(:))];
    Densities(iteration,:) = gather(gpuNormalization)';
    [iteration IterationsMax]
end

save('Densities1080b', 'Densities');
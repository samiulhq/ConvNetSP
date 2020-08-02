function data = customReadDatstoreImage(filename)
% code from default function:
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename); % added lines:
data = data(:,:,min(1:3, end));
data = imresize(data,[128 96]);
end

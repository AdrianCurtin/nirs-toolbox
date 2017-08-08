function hm = measure(this, Hd, varargin)
%MEASURE   Measure the filter.

%   Author(s): J. Schickler
%   Copyright 2005 The MathWorks, Inc.

hm = fdesign.highpassmeas(Hd, this, varargin{:});

% [EOF]
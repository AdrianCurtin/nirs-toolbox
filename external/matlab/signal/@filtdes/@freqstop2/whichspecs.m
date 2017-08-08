function specs = whichspecs(h)
%WHICHSPECS Determine which specs are required for this class.

%   Author(s): R. Losada
%   Copyright 1988-2002 The MathWorks, Inc.

% Prop name, data type, default value, listener callback
specs(1) = cell2struct({'Fstop1','udouble',7200,[],'freqspec'},specfields(h),2);

specs(2) = cell2struct({'Fstop2','udouble',14400,[],'freqspec'},specfields(h),2);


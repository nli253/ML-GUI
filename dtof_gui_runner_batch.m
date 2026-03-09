function dtof_gui_runner_batch
% DTOF ONNX Inference – MATLAB GUI wrapper for onnx_infer.py (BATCH)
% - Choose python.exe, onnx_infer.py, ONNX model
% - Choose INPUT FOLDER containing many *.mat
% - Choose OUTPUT FOLDER to save many output *.mat
% - Set rows (default 51)
% - Run inference for all files and preview a selected file

    fig = uifigure('Name','DTOF ONNX Inference (Batch)','Position',[80 80 1080 640]);

    % ----------- Paths panel -----------
    pnl = uipanel(fig,'Title','Paths','Position',[10 330 1060 300]);
    y = 240;
    [~, edtPython, btnPython]   = row(pnl, 'Python exe:', y);    y=y-40;
    [~, edtInfer,  btnInfer ]   = row(pnl, 'onnx_infer.py:', y); y=y-40;
    [~, edtModel,  btnModel ]   = row(pnl, 'ONNX model:', y);    y=y-40;
    [~, edtInFolder,  btnInFolder ]  = row(pnl, 'Input folder:', y);  y=y-40;
    [~, edtOutFolder, btnOutFolder]  = row(pnl, 'Output folder:', y); y=y-40;

    % Defaults
    edtPython.Value    = 'C:\Users\Diop 3\AppData\Local\Programs\Python\Python37\python.exe';
    edtInfer.Value     = fullfile(pwd,'onnx_infer.py');
    edtModel.Value     = fullfile(pwd,'resendc.onnx');
    edtInFolder.Value  = '';
    edtOutFolder.Value = '';
    global edtPreppedFolder
    global prediction
    
    % ----------- Controls panel -----------
    ctl = uipanel(fig,'Title','Controls','Position',[10 280 1060 50]);

    uilabel(ctl,'Text','Rows (tmp(1:R,:))','Position',[10 6 120 22]);
    edtRows = uieditfield(ctl,'numeric','Position',[135 6 60 22], ...
        'Value',51,'Limits',[1 inf]);

    ddPlot  = uidropdown(ctl,'Items',{'imagesc','mesh'}, ...
        'Value','imagesc','Position',[220 6 100 22]);

    uibutton(ctl,'Text','Prep Data For Inference','Position',[340 6 180 22],...
        'ButtonPushedFcn',@(~,~)prepData());    
    
    uibutton(ctl,'Text','Run Inference (All MAT)','Position',[530 6 180 22],...
        'ButtonPushedFcn',@(~,~)runBatch());

    uibutton(ctl,'Text','Consolidate Output Folder','Position',[720 6 150 22],...
        'ButtonPushedFcn',@(~,~)consolidateData());

    uibutton(ctl,'Text','Quit','Position',[880 6 80 22],...
        'ButtonPushedFcn',@(~,~)close(fig));

    % ----------- Preview area -----------
    ax1 = uiaxes(fig,'Position',[10 10 530 260]); title(ax1,'Input tmp'); colormap(ax1,'parula');
    ax2 = uiaxes(fig,'Position',[545 10 525 260]); title(ax2,'Prediction'); colormap(ax2,'parula');

    uilabel(fig,'Text','Preview output:','Position',[10 590 95 22]);
    ddFiles = uidropdown(fig,'Items',{},'Position',[110 590 520 22],...
        'ValueChangedFcn',@(~,~)previewSelected());

    logBox = uitextarea(fig,'Position',[10 615 1060 40],'Editable','off');
    logBox.Value = "Ready.";

    % Browse callbacks
    btnPython.ButtonPushedFcn    = @(~,~)browseExe(edtPython, 'Select python.exe');
    btnInfer.ButtonPushedFcn     = @(~,~)browseFile(edtInfer,  {'*.py','Python file (*.py)'});
    btnModel.ButtonPushedFcn     = @(~,~)browseFile(edtModel,  {'*.onnx','ONNX model (*.onnx)'});
    btnInFolder.ButtonPushedFcn  = @(~,~)browseFolder(edtInFolder, 'Select input folder with MAT files');
    btnOutFolder.ButtonPushedFcn = @(~,~)browseFolder(edtOutFolder,'Select output folder');

    % ----------- Logger -----------
    function log(msg)
        logBox.Value = [logBox.Value; string(msg)];
        drawnow;
    end

    % ----------- Batch runner -----------
    function runBatch()
        pythonExe = strtrim(edtPython.Value);
        inferPy   = strtrim(edtInfer.Value);
        onnxPath  = strtrim(edtModel.Value);
        inFold    = strtrim(edtPreppedFolder.Value);
        outFold   = strtrim(edtOutFolder.Value);
        rows      = max(1, round(edtRows.Value));

        if ~isfile(pythonExe),  uialert(fig,"Python exe not found","Error");  return; end
        if ~isfile(inferPy),    uialert(fig,"onnx_infer.py not found","Error"); return; end
        if ~isfile(onnxPath),   uialert(fig,"ONNX model not found","Error");  return; end
        if ~isfolder(inFold),   uialert(fig,"Input folder not found","Error"); return; end
        if ~isfolder(outFold),  mkdir(outFold); end

        mats = dir(fullfile(inFold,'*.mat'));
        if isempty(mats)
            uialert(fig,"No .mat files found in input folder.","Info");
            return;
        end

        % Sort by name (if your files have numbers and you want numeric sort,
        % tell me the pattern and I'll adjust)
        [~,ix] = sort({mats.date});
        mats = mats(ix);

        log("Found " + numel(mats) + " MAT files.");
        log("Output folder: " + outFold);

        outItems = strings(0,1);

        for k = 1:numel(mats)
            inMat = fullfile(mats(k).folder, mats(k).name);

            [~, base, ~] = fileparts(mats(k).name);
            outMatName = "pred_" + base + ".mat";
            outMat = fullfile(outFold, outMatName);

            cmd = sprintf('"%s" "%s" -i "%s" -o "%s" -m "%s" --rows %d', ...
                pythonExe, inferPy, inMat, outMat, onnxPath, rows);

            log("[" + k + "/" + numel(mats) + "] " + mats(k).name);
            [st, out] = system(cmd);

            if ~isempty(out)
                log(out);
            end

            if st ~= 0
                log("ERROR: inference failed for " + mats(k).name);
                continue; % keep going
            end

            if isfile(outMat)
                outItems(end+1,1) = outMatName; %#ok<AGROW>
            end
        end

        log("Batch done.");

        if isempty(outItems)
            ddFiles.Items = {};
            ddFiles.Value = '';
            log("No output files were created (all failed?).");
        else
            ddFiles.Items = cellstr(outItems);
            ddFiles.Value = ddFiles.Items{1};
            previewSelected();
        end
    end

    % ----------- Preview selected output -----------
    function previewSelected()
        outFold = strtrim(edtOutFolder.Value);
        inFold  = strtrim(edtInFolder.Value);

        if isempty(ddFiles.Items) || isempty(ddFiles.Value)
            return;
        end

        outMat = fullfile(outFold, ddFiles.Value);

        % Try to infer original input name from pred_ prefix
        name = string(ddFiles.Value);
        inMat = "";
        if startsWith(name,"pred_")
            base = extractAfter(name,"pred_");
            base = erase(base,".mat");
            inMat = fullfile(inFold, base + ".mat");
        end

        % Plot input tmp
        try
            if isfile(inMat)
                M = load(inMat);
                if isfield(M,'tmp')
                    if strcmp(ddPlot.Value,'mesh')
                        mesh(ax1, M.tmp); axis(ax1,'tight'); colorbar(ax1);
                    else
                        imagesc(ax1, M.tmp); axis(ax1,'image'); colorbar(ax1);
                    end
                    title(ax1, sprintf('tmp (%dx%d)', size(M.tmp,1), size(M.tmp,2)));
                else
                    cla(ax1); title(ax1,'tmp not found in input .mat');
                end
            else
                cla(ax1); title(ax1,'Input not found for preview');
            end
        catch ME
            log("Preview input failed: " + ME.message);
        end

        % Plot prediction
        try
            if ~isfile(outMat)
                cla(ax2); title(ax2,'Output .mat not found');
                return;
            end

            S = load(outMat);
            if isfield(S,'prediction')
                P = S.prediction;
                P = flip(P); % your behavior
                if strcmp(ddPlot.Value,'mesh')
                    mesh(ax2, P); axis(ax2,'tight'); colorbar(ax2);
                else
                    imagesc(ax2, P); axis(ax2,'image'); colorbar(ax2);
                end
                title(ax2, sprintf('prediction (%dx%d)', size(P,1), size(P,2)));
            else
                cla(ax2); title(ax2,'prediction not found in output .mat');
            end
        catch ME
            log("Preview prediction failed: " + ME.message);
        end
    end

    function prepData()
        mkdir(strtrim(edtInFolder.Value),'ML 1input');
        edtPreppedFolder.Value = strcat(strtrim(edtInFolder.Value),'\ML 1input');
        inFold = strtrim(edtPreppedFolder.Value);
        
        filePattern = fullfile(strtrim(edtInFolder.Value),'*.txt');
        fid = dir(filePattern);
        [~,inx] = sort({fid.date});
        fid = fid(inx);
        
        dtofs_array = [];
        for i = 2:1:size(fid,1)-1
            [dtofs_array,~] = LoadDTOFs([strtrim(edtInFolder.Value),'\',fid(i).name],[strtrim(edtInFolder.Value),'\',fid(1).name],dtofs_array,'SPAD');
        end   
        
        ddtofs = zeros(size(dtofs_array));
        for i = 1:size(dtofs_array,1)
            ddtofs(i,:) = denoised(dtofs_array(i,:)); 
        end
        
        rows_per_block = 104;
        num_blocks = size(ddtofs,1) / rows_per_block;

        for k = 1:num_blocks
            row_start = (k-1)*rows_per_block + 1;
            row_end   = k*rows_per_block;
    
            block = ddtofs(row_start:row_end, :);   % 104 × 820
            tmp=block(1:2:end,:)-block(2:2:end,:);

            output_file_name = fullfile(inFold, sprintf('DTOF_%d.mat', k));
            save(output_file_name,'tmp');
        end
    end

    function consolidateData()
        filePattern = fullfile(strtrim(edtOutFolder.Value),'*.mat');
        fid = dir(filePattern);
        [~,inx] = sort({fid.date});
        fid = fid(inx);
        
        for n = 1:size(fid,1)
            load([strtrim(edtOutFolder.Value),'\',fid(n).name])
            DTOF(n).a =  prediction';
        end
        
        delete(filePattern)
        output_file_name = fullfile(edtOutFolder.Value, sprintf('DTOF_ML.mat'));
        save(output_file_name,'DTOF') 
    end

    % ----------- Small helpers -----------
    function [lab, edt, btn] = row(parent, label, y)
        lab = uilabel(parent,'Text',label,'Position',[10 y 110 22]);
        edt = uieditfield(parent,'text','Position',[125 y 825 22],'Value','');
        btn = uibutton(parent,'Text','Browse','Position',[960 y 75 22]);
    end

    function browseExe(edt, prompt)
        [f,p] = uigetfile({'*.exe','Executable (*.exe)'}, prompt);
        if isequal(f,0) || isequal(p,0), return; end
        edt.Value = fullfile(p,f);
    end

    function browseFile(edt, filter)
        [f,p] = uigetfile(filter);
        if isequal(f,0) || isequal(p,0), return; end
        edt.Value = fullfile(p,f);
    end

    function browseFolder(edt, prompt)
        p = uigetdir(edt.Value, prompt);
        if isequal(p,0), return; end
        edt.Value = p;
    end
end

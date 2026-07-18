# Mesh QC Review Tool

This tool displays one tissue-render image at a time in your normal web browser.
Everything runs locally on the computer. It does not upload images or decisions,
and it does not require an internet connection after Python has been installed.

## Before Starting On macOS

1. Extract the complete ZIP archive to a writable location such as Documents or
   Desktop. Do not run the tool from inside the ZIP preview.
2. Confirm that the supplied render folders are inside the `images` folder.
3. Confirm that Python 3.10 or newer is installed by opening Terminal and running
   `python3 --version`. Current Python installers are available from
   <https://www.python.org/downloads/macos/>.
4. Keep the tool folder in the same location until the review is complete.
   Moving it is possible, but the `images` and `review_state` folders must stay
   together with the application files.

No Python packages need to be installed. The application uses only components
included with Python.

## Starting The Tool

Double-click `Start Mesh Review.command`.

The Terminal window will remain open and the review page will open in the
default browser. Keep the Terminal window open while reviewing. The exact local
address is printed there and normally resembles `http://127.0.0.1:54321/`.

If macOS refuses to open the launcher:

1. Control-click `Start Mesh Review.command` and choose **Open**.
2. If that is still blocked, open Terminal, type `bash ` including the trailing
   space, drag `Start Mesh Review.command` into the Terminal window, and press
   Return.

## Review Rules

- **Accept** means the displayed tissue render is suitable for inclusion.
- **Maybe** defers the image to the separate `Revisit maybe` queue.
- **Decline subject** excludes the entire subject. Once any image is declined,
  all remaining images for that subject disappear from the active queue.
- A subject becomes fully accepted only after every discovered image for that
  subject has been accepted.
- The tool alerts you when the accepted-subject count reaches 200.

Keyboard shortcuts are `A` for Accept, `M` for Maybe, and `D` for Decline. They
are disabled while typing in the notes field. **Undo** restores the immediately
preceding decision.

Use **Subject first** to inspect all tissues/views for one subject together. Use
**Tissue first** to compare the same tissue across subjects. The Subjects tab
shows aggregate progress and supports subject-ID search.

## Saving And Resuming

Every decision is written immediately to:

```text
review_state/mesh_review.sqlite3
```

To stop, return to the Terminal window and press Control-C once. The tool then
refreshes all exports before closing. Start it again later with the same launcher
to resume exactly where you stopped.

Do not delete or rename images after starting a review. If additional images are
added, use **Rescan folder**. A newly added image reopens an accepted subject
until that image is reviewed.

## Returning The Results

Use **Export** in the browser, then stop the tool with Control-C. Return the
complete `review_state` folder. Its human-readable outputs are:

```text
review_state/exports/accepted_subjects.txt
review_state/exports/maybe_subjects.txt
review_state/exports/declined_subjects.txt
review_state/exports/subject_summary.csv
review_state/exports/image_decisions.csv
review_state/exports/review_manifest.json
```

The SQLite database is the authoritative record, so returning the complete
folder is preferable to returning only `accepted_subjects.txt`.

## Troubleshooting

- Browser page did not open: copy the local `http://127.0.0.1:.../` address from
  Terminal into Safari, Chrome, or Firefox.
- Zero images discovered: ensure the render folders are under `images` and that
  filenames contain IDs such as `sub-CC120001`.
- Images were added later: click **Rescan folder**.
- Launcher reports an old Python: install a current Python 3 release and restart
  Terminal.
- The browser was closed accidentally: reopen the local address printed in the
  still-running Terminal window. Progress has not been lost.

import { Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import * as ort from 'onnxruntime-web';
import {  from, interval, map, Subscription, switchMap, tap } from 'rxjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnDestroy {
  @ViewChild("canvas")
  canvas!: ElementRef<HTMLCanvasElement>;

  sub: Subscription;
  constructor() {
    this.sub = from(ort.InferenceSession.create('./model.onnx')).pipe(
      switchMap(session => interval(50).pipe(map(_ => session))),
      switchMap(session => {
        const date = new Date();
        const inputArray = new Int32Array([
          Math.floor(date.getHours() / 10), 
          date.getHours() % 10, 
          11, 
          Math.floor(date.getMinutes() / 10),
          date.getMinutes() % 10, 
          11, 
          Math.floor(date.getSeconds() / 10),
          date.getSeconds() % 10,
        ]);
        const input = new ort.Tensor('int32', inputArray, [inputArray.length]);
        const inputName = session.inputNames[0];
        const feeds: Record<string, ort.Tensor> = {
          [inputName]: input
        };
        return from(session.run(feeds)).pipe(
          map(output => [session, output] as const)
        );
      }),
      switchMap(([session, output]) => {
        const tensor = output[session.outputNames[0]];
        const ctx = this.canvas.nativeElement.getContext("2d");
        if (!ctx) {
          throw new Error("Could not get 2d context");
        }
        const tensorData = tensor.data as Int32Array;
        const imageData = new ImageData(new Uint8ClampedArray(tensorData), 28 * tensor.dims[1], 28)
        return from(createImageBitmap(imageData));
      }),
      tap(bitmap => {
        const ctx = this.canvas.nativeElement.getContext("2d");
        if (!ctx) {
          throw new Error("Could not get 2d context");
        }
        ctx.fillRect(0, 0, bitmap.width * 4, bitmap.height * 4)
        ctx.drawImage(bitmap, 0, 0, bitmap.width * 4, bitmap.height * 4);
        ctx.fillRect(28 * 4 * 2, 0, 28 * 4, 28 * 4)
        ctx.fillRect(28 * 4 * 5, 0, 28 * 4, 28 * 4)
        bitmap.close()
      })
    ).subscribe();
  }

  ngOnDestroy(): void {
  }
}

build-image /home/AD/npaneru/deploy_repo/dos-ai/Dockerfile reg.dcaic.deloitte.com/dos-gen-ai/style:22-demo
kubectl delete deployment stylingapp
kubectl delete service stylingapp-service
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get service